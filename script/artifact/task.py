import time
import threading
import paramiko
import subprocess

from .gcp.gcp_utils import *
from .ssh_utils import *

ZONE = "us-central1-a"
VPC_NAME = "skypilot-vpc"
IMAGE_NAME = "artifact"
PROJECT_ID = "encrypted-search-404320"
PRIVATE_KEY_PATH = os.path.expanduser("~/.ssh/compass_artifact")
USER_NAME = "artifact"

throuput = []

client_config = {
    # "name": "ae-client",
    "machine_type": "n2-standard-8",
    "boot_disk_image": f"projects/{PROJECT_ID}/global/images/{IMAGE_NAME}",
    "disk_size_gb": 512,  # Optional, defaults to 10GB
    "disk_type" :f"zones/{ZONE}/diskTypes/pd-ssd",
    "zone": ZONE,
    "vpc_name": VPC_NAME,
    "project_id": PROJECT_ID

}

server_config = {
    # "name": "ae-server",
    "machine_type": "n2-highmem-64",
    "boot_disk_image": f"projects/{PROJECT_ID}/global/images/{IMAGE_NAME}",
    "disk_size_gb": 512,  # Optional, defaults to 10GB
    "disk_type" :f"zones/{ZONE}/diskTypes/pd-ssd",
    "zone": ZONE,
    "vpc_name": VPC_NAME,
    "project_id": PROJECT_ID
}

compass_datasets_sh = ["laion", "sift", "trip", "msmarco"]
compass_datasets = ["laion", "laion_mal", "sift", "sift_mal", "trip", "trip_mal", "msmarco", "msmarco_mal"]
# compass_datasets = ["laion"]

def check_ssh_instances(instances):
    threads = []

    print("-> Wait for instances to be ready")

    for instance in instances:

        thread = threading.Thread(target=check_ssh_connection, args=(instance["internal_ip"], USER_NAME, PRIVATE_KEY_PATH, True))

        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

def prepare_instance(instance):
    print("-> Prepare instance:", instance["name"])
    cmds = [
    "ps aux | grep compass | grep -v grep | awk '{print $2}' | xargs kill",
    "cd /home/artifact/compass/ && git pull",
    "cd /home/artifact/compass/build/ && make"
    ]

    execute_commands_queit(
        instance["name"],
        instance["internal_ip"],
        cmds,
        PRIVATE_KEY_PATH,
        USER_NAME,
        False
    )

def prepare_instances(instances):

    print("-> Prepare instances")
    threads = []
    for instance in instances:
        thread = threading.Thread(target=prepare_instance, args=(instance, ))
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()


def tc_reset(instance):
    cmds = [
    "sudo setcap cap_net_admin+ep /usr/sbin/tc",
    "/home/artifact/.local/bin/tcdel ens4 --all"
    ]

    execute_commands_queit(
        instance["name"],
        instance["internal_ip"],
        cmds,
        PRIVATE_KEY_PATH,
        USER_NAME,
        True
    )


def tc_slow(instance):
    cmds = [
    "/home/artifact/.local/bin/tcset ens4 --delay 40ms --rate 400Mbps"
    ]

    execute_commands_queit(
        instance["name"],
        instance["internal_ip"],
        cmds,
        PRIVATE_KEY_PATH,
        USER_NAME,
        True
    )


def tc_fast(instance):
    cmds = [
    "/home/artifact/.local/bin/tcset ens4 --delay 0.5ms --rate 3Gbps"
    ]

    execute_commands_queit(
        instance["name"],
        instance["internal_ip"],
        cmds,
        PRIVATE_KEY_PATH,
        USER_NAME,
        True
    )

def tc_show(instance):
    cmds = [
    "/home/artifact/.local/bin/tcshow ens4"
    ]

    execute_commands_queit(
        instance["name"],
        instance["internal_ip"],
        cmds,
        PRIVATE_KEY_PATH,
        USER_NAME,
        True
    )

# accuracy only requires one instance
def run_compass_accuracy(instance, verbose):

    print("Exp: compass accuracy ")

    # prepare instance
    # print("Preparing instance...")
    prepare_instance(instance)

    print("-> Run exp...")
    # run accuracy exp 
    # client_cmds = []
    # server_cmds = []

    accuracy_file_list = []

    for d in compass_datasets_sh:
        print("-> dataset:", d)
        f_accuracy = "accuracy_" + d + ".ivecs"
        accuracy_file_list.append(f_accuracy)

        s = "cd compass/build/ && " + f"./test_compass_accuracy r=1 d={d}"
        c = "cd compass/build/ && " + f"./test_compass_accuracy r=2 d={d} f_accuracy={f_accuracy}"
        
        # client_cmds.append(s)
        # server_cmds.append(c)

        threads = []

        s_thread = threading.Thread(target=execute_commands_queit, args=(instance["name"], instance["internal_ip"], [s], PRIVATE_KEY_PATH, USER_NAME, False))
        c_thread = threading.Thread(target=execute_commands_queit, args=(instance["name"], instance["internal_ip"], [c], PRIVATE_KEY_PATH, USER_NAME, verbose))
        
        threads.append(s_thread)
        threads.append(c_thread)
        s_thread.start()
        c_thread.start()

        for thread in threads:
            thread.join()
        
    # retrieve result
    remote_fpath_list = ["/home/artifact/compass/build/" + f for f in accuracy_file_list]

    print("-> Fetch results...")
    scp_from_remote(
        instance["internal_ip"],
        USER_NAME,
        PRIVATE_KEY_PATH,
        remote_fpath_list, 
        "./script/artifact/results/"
    )

    print("-> Done.")

    return 

def run_compass_latency(server_instance, client_instance, verbose):

    print("Exp: compass latency ")

    n = 100
    # prepare instance
    # print("Preparing instance...")
    prepare_instances([server_instance, client_instance])

    remote_fpath_list = []

    print("-> Run exp...")
    # fast network
    tc_reset(server_instance)
    tc_reset(client_instance)
    tc_fast(server_instance)
    tc_fast(client_instance)
    print("-> Fast network...")

    for d in compass_datasets:
        print("-> dataset:", d)
        f_latency = "latency_fast_" + d + ".fvecs"
        f_comm = "comm_" + d + ".txt"
        remote_fpath_list.append(f_latency)
        remote_fpath_list.append(f_comm)

        s = "cd compass/build/ && " + f"./test_compass_ring r=1 d={d} n={n} ip={server_instance['internal_ip']} f_comm={f_comm}"
        c = "cd compass/build/ && " + f"./test_compass_ring r=2 d={d} n={n} ip={server_instance['internal_ip']} f_comm={f_comm} f_latency={f_latency}"
        
        # client_cmds.append(s)
        # server_cmds.append(c)

        threads = []

        s_thread = threading.Thread(target=execute_commands_queit, args=(server_instance["name"], server_instance["internal_ip"], [s], PRIVATE_KEY_PATH, USER_NAME, False))
        c_thread = threading.Thread(target=execute_commands_queit, args=(client_instance["name"], client_instance["internal_ip"], [c], PRIVATE_KEY_PATH, USER_NAME, verbose))
        
        threads.append(s_thread)
        threads.append(c_thread)
        s_thread.start()
        c_thread.start()

        for thread in threads:
            thread.join()
    

    # slow network
    tc_reset(server_instance)
    tc_reset(client_instance)
    tc_slow(server_instance)
    tc_slow(client_instance)
    print("-> Slow network...")

    for d in compass_datasets:
        print("-> dataset:", d)
        f_latency = "latency_slow_" + d + ".fvecs"
        remote_fpath_list.append(f_latency)

        s = "cd compass/build/ && " + f"./test_compass_ring r=1 d={d} n={n} ip={server_instance['internal_ip']}"
        c = "cd compass/build/ && " + f"./test_compass_ring r=2 d={d} n={n} ip={server_instance['internal_ip']} f_latency={f_latency}"
        
        # client_cmds.append(s)
        # server_cmds.append(c)

        threads = []

        s_thread = threading.Thread(target=execute_commands_queit, args=(server_instance["name"], server_instance["internal_ip"], [s], PRIVATE_KEY_PATH, USER_NAME, False))
        c_thread = threading.Thread(target=execute_commands_queit, args=(client_instance["name"], client_instance["internal_ip"], [c], PRIVATE_KEY_PATH, USER_NAME, verbose))
        
        threads.append(s_thread)
        threads.append(c_thread)
        s_thread.start()
        c_thread.start()

        for thread in threads:
            thread.join()
    

    print("-> Fetch results...")

    remote_fpath_list = ["/home/artifact/compass/build/" + f for f in remote_fpath_list]

    scp_from_remote(
        client_instance["internal_ip"],
        USER_NAME,
        PRIVATE_KEY_PATH,
        remote_fpath_list, 
        "./script/artifact/results/"
    )

    print("-> Done.")

    return



def run_baseline_latency(server_instance, client_instance, verbose):

    print("Exp: baseline latency ")

    # prepare instance
    # print("Preparing instance...")
    prepare_instances([server_instance, client_instance])

    remote_fpath_list = []

    print("-> Run exp...")
    # fast network
    tc_reset(server_instance)
    tc_reset(client_instance)
    tc_fast(server_instance)
    tc_fast(client_instance)
    print("-> Fast network")

    # obi
    n = 10
    for d in ["trip", "msmarco"]:
        for trunc in [10, 100, 1000, 10000]:
            print(f"-> Inv-ORAM: {d} truncation {trunc}")
            f_latency = f"latency_obi_fast_{d}_{trunc}.fvecs"
            f_comm = f"comm_obi_{d}_{trunc}.txt"

            remote_fpath_list.append(f_latency)
            remote_fpath_list.append(f_comm)

            s = "cd compass/build/ && " + f"./test_obi r=1 d={d} n={n} trunc={trunc} ip={server_instance['internal_ip']} f_comm={f_comm}"
            c = "cd compass/build/ && " + f"./test_obi r=2 d={d} n={n} trunc={trunc} ip={server_instance['internal_ip']} f_comm={f_comm} f_latency={f_latency}"

            threads = []

            s_thread = threading.Thread(target=execute_commands_queit, args=(server_instance["name"], server_instance["internal_ip"], [s], PRIVATE_KEY_PATH, USER_NAME, False))
            c_thread = threading.Thread(target=execute_commands_queit, args=(client_instance["name"], client_instance["internal_ip"], [c], PRIVATE_KEY_PATH, USER_NAME, verbose))
            
            threads.append(s_thread)
            threads.append(c_thread)
            s_thread.start()
            c_thread.start()

            for thread in threads:
                thread.join()

    
    # cluster search
    for d in ["laion", "sift", "trip"]:
        print("-> HE-Cluster:", d)
        f_latency = f"latency_cluster_fast_{d}.fvecs"
        f_comm = f"comm_cluster_{d}.txt"

        remote_fpath_list.append(f_latency)
        remote_fpath_list.append(f_comm)

        s = "cd compass/build/ && " + f"./test_cluster_search r=1 d={d} ip={server_instance['internal_ip']} f_comm={f_comm}"
        c = "cd compass/build/ && " + f"./test_cluster_search r=2 d={d} ip={server_instance['internal_ip']} f_comm={f_comm} f_latency={f_latency}"

        threads = []

        s_thread = threading.Thread(target=execute_commands_queit, args=(server_instance["name"], server_instance["internal_ip"], [s], PRIVATE_KEY_PATH, USER_NAME, False))
        c_thread = threading.Thread(target=execute_commands_queit, args=(client_instance["name"], client_instance["internal_ip"], [c], PRIVATE_KEY_PATH, USER_NAME, verbose))
        
        threads.append(s_thread)
        threads.append(c_thread)
        s_thread.start()
        c_thread.start()

        for thread in threads:
            thread.join()

    # fast network
    tc_reset(server_instance)
    tc_reset(client_instance)
    tc_slow(server_instance)
    tc_slow(client_instance)
    print("-> Slow network")

    n = 10
    for d in ["trip", "msmarco"]:
        for trunc in [10, 100, 1000, 10000]:
            print(f"-> Inv-ORAM: {d} truncation {trunc}")
            f_latency = f"latency_obi_slow_{d}_{trunc}.fvecs"

            remote_fpath_list.append(f_latency)
            remote_fpath_list.append(f_comm)

            s = "cd compass/build/ && " + f"./test_obi r=1 d={d} n={n} trunc={trunc} ip={server_instance['internal_ip']}"
            c = "cd compass/build/ && " + f"./test_obi r=2 d={d} n={n} trunc={trunc} ip={server_instance['internal_ip']} f_latency={f_latency}"

            threads = []

            s_thread = threading.Thread(target=execute_commands_queit, args=(server_instance["name"], server_instance["internal_ip"], [s], PRIVATE_KEY_PATH, USER_NAME, False))
            c_thread = threading.Thread(target=execute_commands_queit, args=(client_instance["name"], client_instance["internal_ip"], [c], PRIVATE_KEY_PATH, USER_NAME, verbose))
            
            threads.append(s_thread)
            threads.append(c_thread)
            s_thread.start()
            c_thread.start()

            for thread in threads:
                thread.join()

    
    # cluster search
    for d in ["laion", "sift", "trip"]:
        print("-> HE-Cluster:", d)
        f_latency = f"latency_cluster_slow_{d}.fvecs"
        f_comm = f"comm_cluster_{d}.fvecs"

        remote_fpath_list.append(f_latency)
        remote_fpath_list.append(f_comm)

        s = "cd compass/build/ && " + f"./test_cluster_search r=1 d={d} ip={server_instance['internal_ip']} f_comm={f_comm}"
        c = "cd compass/build/ && " + f"./test_cluster_search r=2 d={d} ip={server_instance['internal_ip']} f_comm={f_comm} f_latency={f_latency}"

        threads = []

        s_thread = threading.Thread(target=execute_commands_queit, args=(server_instance["name"], server_instance["internal_ip"], [s], PRIVATE_KEY_PATH, USER_NAME, False))
        c_thread = threading.Thread(target=execute_commands_queit, args=(client_instance["name"], client_instance["internal_ip"], [c], PRIVATE_KEY_PATH, USER_NAME, verbose))
        
        threads.append(s_thread)
        threads.append(c_thread)
        s_thread.start()
        c_thread.start()

        for thread in threads:
            thread.join()

    print("-> Fetch results...")

    remote_fpath_list = ["/home/artifact/compass/build/" + f for f in remote_fpath_list]

    scp_from_remote(
        client_instance["internal_ip"],
        USER_NAME,
        PRIVATE_KEY_PATH,
        remote_fpath_list, 
        "./script/artifact/results/"
    )

    print("-> Done.")

    return

def run_ablation_accuracy(instance, verbose):

    d = "msmarco" 

    # prepare instance
    print("Preparing instance...")
    prepare_instance(instance)

    print("Run exp...")
    s_cmds = []
    c_cmds = []

    accuracy_file_list = []
    
    # fix efn
    efn = 24
    efspec = 1
    while efspec <= 16:
        f_accuracy  = f"ablation_accuracy_{d}_{efspec}_{efn}.ivecs"
        accuracy_file_list.append(f_accuracy)
        s = "cd compass/build/ && " + f"./test_compass_accuracy r=1 d={d} efn={efn} efspec={efspec}"
        c = "cd compass/build/ && " + f"./test_compass_accuracy r=2 d={d} efn={efn} efspec={efspec} f_accuracy={f_accuracy}"
        s_cmds.append(s)
        c_cmds.append(c)
        efspec = efspec*2

    # fix efspec
    efspec = 8
    efn = 1
    while efn <= 256:
        f_accuracy  = f"ablation_accuracy_{d}_{efspec}_{efn}.ivecs"
        accuracy_file_list.append(f_accuracy)
        s = "cd compass/build/ && " + f"./test_compass_accuracy r=1 d={d} efn={efn} efspec={efspec}"
        c = "cd compass/build/ && " + f"./test_compass_accuracy r=2 d={d} efn={efn} efspec={efspec} f_accuracy={f_accuracy}"
        s_cmds.append(s)
        c_cmds.append(c)
        efn = efn*2

    # run accuracy
    for s, c in zip(s_cmds, c_cmds):

        threads = []

        s_thread = threading.Thread(target=execute_commands_queit, args=(instance["name"], instance["internal_ip"], [s], PRIVATE_KEY_PATH, USER_NAME, False))
        c_thread = threading.Thread(target=execute_commands_queit, args=(instance["name"], instance["internal_ip"], [c], PRIVATE_KEY_PATH, USER_NAME, verbose))
        
        threads.append(s_thread)
        threads.append(c_thread)
        s_thread.start()
        c_thread.start()

        for thread in threads:
            thread.join()

    # fetch results
    remote_fpath_list = ["/home/artifact/compass/build/" + f for f in accuracy_file_list]


    scp_from_remote(
        instance["internal_ip"],
        USER_NAME,
        PRIVATE_KEY_PATH,
        remote_fpath_list, 
        "./script/artifact/results/"
    )
    
    return 

def run_ablation_accuracy_optimized(instance, verbose):

    d = "msmarco" 

    # prepare instance
    print("Preparing instance...")
    prepare_instance(instance)

    print("Run exp...")
    # s_cmds = []
    # c_cmds = []

    accuracy_file_list = []
    
    # fix efn
    efn = 24
    efspec = 1
    while efspec <= 16:
        f_accuracy  = f"ablation_accuracy_{d}_{efspec}_{efn}.ivecs"
        accuracy_file_list.append(f_accuracy)
        # s = "cd compass/build/ && " + f"./test_compass_accuracy r=1 d={d} efn={efn} efspec={efspec}"
        # c = "cd compass/build/ && " + f"./test_compass_accuracy r=2 d={d} efn={efn} efspec={efspec} f_accuracy={f_accuracy}"
        # s_cmds.append(s)
        # c_cmds.append(c)
        efspec = efspec*2

    # fix efspec
    efspec = 8
    efn = 1
    while efn <= 256:
        f_accuracy  = f"ablation_accuracy_{d}_{efspec}_{efn}.ivecs"
        accuracy_file_list.append(f_accuracy)
        # s = "cd compass/build/ && " + f"./test_compass_accuracy r=1 d={d} efn={efn} efspec={efspec}"
        # c = "cd compass/build/ && " + f"./test_compass_accuracy r=2 d={d} efn={efn} efspec={efspec} f_accuracy={f_accuracy}"
        # s_cmds.append(s)
        # c_cmds.append(c)
        efn = efn*2

    # run accuracy
    # for s, c in zip(s_cmds, c_cmds):

    threads = []

    s = "cd compass/build/ && " + f"./test_compass_accuracy_ablation r=1 d={d} max_efn={256} max_efspec={16}"
    c = "cd compass/build/ && " + f"./test_compass_accuracy_ablation r=2 d={d} max_efn={256} max_efspec={16}"

    s_thread = threading.Thread(target=execute_commands_queit, args=(instance["name"], instance["internal_ip"], [s], PRIVATE_KEY_PATH, USER_NAME, False))
    c_thread = threading.Thread(target=execute_commands_queit, args=(instance["name"], instance["internal_ip"], [c], PRIVATE_KEY_PATH, USER_NAME, verbose))
    
    threads.append(s_thread)
    threads.append(c_thread)
    s_thread.start()
    c_thread.start()

    for thread in threads:
        thread.join()

    # fetch results
    remote_fpath_list = ["/home/artifact/compass/build/" + f for f in accuracy_file_list]


    scp_from_remote(
        instance["internal_ip"],
        USER_NAME,
        PRIVATE_KEY_PATH,
        remote_fpath_list, 
        "./script/artifact/results/"
    )
    
    return 


def run_ablation_latency(server_instance, client_instance, verbose):

    n = 10
    d="msmarco"

    # prepare instance
    print("Preparing instance...")
    prepare_instances([server_instance, client_instance])

    # slow network
    tc_reset(server_instance)
    tc_reset(client_instance)
    tc_slow(server_instance)
    tc_slow(client_instance)

    # prepare cmds
    latency_file_list = []
    s_cmds = []
    c_cmds = []

     # fix efn
    efn = 24
    efspec = 1
    while efspec <= 16:
        f_latency  = f"ablation_latency_{d}_{efspec}_{efn}.fvecs"
        latency_file_list.append(f_latency)
        s = "cd compass/build/ && " + f"./test_compass_ring r=1 d={d} n={n} efn={efn} efspec={efspec} ip={server_instance['internal_ip']}"
        c = "cd compass/build/ && " + f"./test_compass_ring r=2 d={d} n={n} efn={efn} efspec={efspec} ip={server_instance['internal_ip']} f_latency={f_latency}"
        s_cmds.append(s)
        c_cmds.append(c)
        efspec = efspec*2

    # fix efspec
    efspec = 8
    efn = 1
    while efn <= 256:
        f_latency  = f"ablation_latency_{d}_{efspec}_{efn}.fvecs"
        latency_file_list.append(f_latency)
        s = "cd compass/build/ && " + f"./test_compass_ring r=1 d={d} n={n} efn={efn} efspec={efspec} ip={server_instance['internal_ip']}"
        c = "cd compass/build/ && " + f"./test_compass_ring r=2 d={d} n={n} efn={efn} efspec={efspec} ip={server_instance['internal_ip']} f_latency={f_latency}"
        s_cmds.append(s)
        c_cmds.append(c)
        efn = efn*2
    
    # disable lazy_eviction
    f_latency  = f"ablation_latency_{d}_lazy.fvecs"
    latency_file_list.append(f_latency)
    s_lazy = "cd compass/build/ && " + f"./test_compass_ring r=1 d={d} n={n} lazy=0 ip={server_instance['internal_ip']}"
    c_lazy = "cd compass/build/ && " + f"./test_compass_ring r=2 d={d} n={n} lazy=0 ip={server_instance['internal_ip']} f_latency={f_latency}"
    s_cmds.append(s_lazy)
    c_cmds.append(c_lazy)

    # vanilla ring oram
    f_latency  = f"ablation_latency_{d}_vanilla.fvecs"
    latency_file_list.append(f_latency)
    s_vanilla = "cd compass/build/ && " + f"./test_compass_ring r=1 d={d} n={n} lazy=0 batch=0 ip={server_instance['internal_ip']}"
    c_vanilla = "cd compass/build/ && " + f"./test_compass_ring r=2 d={d} n={n} lazy=0 batch=0 ip={server_instance['internal_ip']} f_latency={f_latency}"
    s_cmds.append(s_vanilla)
    c_cmds.append(c_vanilla)

    # run accuracy
    for s, c in zip(s_cmds, c_cmds):

        threads = []

        s_thread = threading.Thread(target=execute_commands_queit, args=(server_instance["name"], server_instance["internal_ip"], [s], PRIVATE_KEY_PATH, USER_NAME, False))
        c_thread = threading.Thread(target=execute_commands_queit, args=(client_instance["name"], client_instance["internal_ip"], [c], PRIVATE_KEY_PATH, USER_NAME, verbose))
        
        threads.append(s_thread)
        threads.append(c_thread)
        s_thread.start()
        c_thread.start()

        for thread in threads:
            thread.join()

    # fetch results
    remote_fpath_list = ["/home/artifact/compass/build/" + f for f in latency_file_list]


    scp_from_remote(
        client_instance["internal_ip"],
        USER_NAME,
        PRIVATE_KEY_PATH,
        remote_fpath_list, 
        "./script/artifact/results/"
    )

    return

def execute_commands_reduce(instance_name, ip, cmds, private_key_path, user_name, verbose, log_tp):

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    private_key = paramiko.Ed25519Key.from_private_key_file(private_key_path)

    if log_tp:
        id = int(instance_name.split('-')[-1])

    try:
        if verbose:
            print(f"Try to connect to: {instance_name}")

        ssh_client.connect(hostname=ip, username=user_name, pkey=private_key)
        if verbose:
            print(f"Connected to {instance_name} at {ip}")

        for cmd in cmds:

            if verbose:
                print("exec_command:", cmd)

            transport = ssh_client.get_transport()
            channel = transport.open_session()
            channel.exec_command(cmd)

            if verbose:
                # Read the stdout in real-time
                while True:
                    # Check if data is ready to be read
                    if channel.recv_ready():
                        output = channel.recv(1024).decode('utf-8')  # Adjust buffer size if necessary
                        print(f"[{instance_name}]:", output, end="")  # Print the output without adding extra newlines
                    
                    # Check if the command is finished
                    if channel.exit_status_ready():
                        output = channel.recv(1024).decode('utf-8')  # Adjust buffer size if necessary
                        print(f"[{instance_name}]:",output, end="")  # Print the output without adding extra newlines
                        break

                    # Prevent high CPU usage in the loop
                    time.sleep(0.1)
            else:
                # wait until the cmd is complete
                while True:
                    # Check if data is ready to be read
                    if log_tp:
                        if channel.recv_ready():
                            output = channel.recv(1024).decode('utf-8')  # Adjust buffer size if necessary
                            if output.split(" ")[0] == "Throughput:":
                                tp = float(output.split(" ")[1])
                                # print("assign to id:", id, " length of trhouput:", len(throuput))
                                # print(throuput)
                                throuput[id] = tp
                    
                    # Check if the command is finished
                    if channel.exit_status_ready():
                        # output = channel.recv(1024).decode('utf-8')  # Adjust buffer size if necessary
                        # print(f"[{instance_name}]:",output, end="")  # Print the output without adding extra newlines
                        break

                    # Prevent high CPU usage in the loop
                    time.sleep(0.1)

        if verbose:
            print("\n")

        # Close the connection
        ssh_client.close()
    except Exception as e:
        print(f"Error: {e}")


def tp_monitor():
    while True:
        total_tp = 0
        for tp in throuput:
            total_tp += tp
        print(f"Total throughput: {total_tp:.3f}")
        time.sleep(1)

def run_throughput(server_instance, client_instances, verbose):

    # prepare_instance(server_instance)
    # for instance in client_instances:
    #     prepare_instance(instance)

    prepare_instances([server_instance] + client_instances)

    dataset = "laion"
    port = 9000
    n_clients = len(client_instances)
    server_ip = server_instance["internal_ip"]

    # monitor
    global throuput
    throuput = [0] * len(client_instances)
    print(throuput)

    threads = []

    # server
    cmd = f"cd compass/script/artifact/throughput/ && ./run_server.sh {n_clients} {server_ip}"
    s_thread = threading.Thread(target=execute_commands_queit, args=(server_instance["name"], server_instance["internal_ip"], [cmd], PRIVATE_KEY_PATH, USER_NAME, False))
    threads.append(s_thread)
    s_thread.start()

    # clients
    cmds = []
    for i in range(n_clients):
        cmd = f"cd compass/build/  && ./test_compass_tp r=2 ip={server_ip} p={port+2*i} d={dataset}"
        cmds.append(cmd)

    reset_cmd = "ps aux | grep compass | grep -v grep | awk '{print $2}' | xargs kill"

    for tracer, cmd in zip(client_instances, cmds):
        thread = threading.Thread(target=execute_commands_reduce, args=(tracer["name"], tracer["internal_ip"], [reset_cmd, cmd], PRIVATE_KEY_PATH, USER_NAME, False, True))
        threads.append(thread)
        thread.start()

    monitor_thread = threading.Thread(target=tp_monitor, args=())
    threads.append(monitor_thread)
    monitor_thread.start()

    for thread in threads:
        thread.join()


def stop_instances():
    instances = get_instances(PROJECT_ID)
    ae_instance_names = []
    for instance in instances:
        if "ae-" in instance["name"]:
            ae_instance_names.append(instance["name"])

    print("ae instances:", ae_instance_names)
    
    delete_tracers(PROJECT_ID, ZONE, ae_instance_names)

if __name__ == "__main__":
    os.chdir(os.path.expanduser('~/compass/'))
   
    # (c_name, c_internal_ip) = create_instance(client_config, "ae-client")

    # c_instance = {
    #     "name": "ae-client",
    #     "internal_ip": "10.128.0.31"
    # }

    # s_instance = {
    #     "name": "ae-server",
    #     "internal_ip": "10.128.0.30"
    # }

    # delete_instance(PROJECT_ID, ZONE, "ae-client")
    # delete_instance(PROJECT_ID, ZONE, "ae-server")


    # tracer_instance_1 = {
    #     "name": "ae-throughput-tracer-0",
    #     "internal_ip": "10.128.0.40"
    # }

    # tracer_instance_2 = {
    #     "name": "ae-throughput-tracer-1",
    #     "internal_ip": "10.128.0.41"
    # }

    # tracers = [tracer_instance_1, tracer_instance_2]


    # run_throuput(s_instance, tracers)

    # tracer_names = [tracer["name"] for tracer in tracers]
    # delete_tracers(PROJECT_ID, ZONE, tracer_names)

    # tracer_names = ['ae-throughput-tracer-0', 'ae-throughput-tracer-1', 'ae-throughput-tracer-2', 'ae-throughput-tracer-3', 'ae-throughput-tracer-4']
    # delete_tracers(PROJECT_ID, ZONE, tracer_names)

    # print(tracer_names)

    # run_baseline_latency(s_instance, c_instance)

    # run_ablation_accuracy(s_instance)
    # run_ablation_latency(s_instance, c_instance)

    # prepare_instance(s_instance)
    # prepare_instance(c_instance)

    # run_compass_latency(s_instance, c_instance)
    # tc_slow(s_instance)

    # prepare_instance(s_instance)
    # prepare_instance(c_instance)


    # instance = {
    #     "name": sname,
    #     "internal_ip": internal_ip
    # }

    # private_key = paramiko.RSAKey.from_private_key_file(PRIVATE_KEY_PATH)

    # instance = {
    #     "name": "ae_server",
    #     "internal_ip": "10.128.0.26"
    # }
    # run_compass_accuracy(instance)
    # sname = "ae-server"
    # delete_instance(PROJECT_ID, ZONE, sname)

