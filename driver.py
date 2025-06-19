import argparse
import subprocess
import sys
import time
import os

from script.artifact.task import *
# from script.artifact.graph.latency_breakdown import render_latency_breakdown
# from script.artifact.graph.ablation_study import render_msmarco_ablation
from script.artifact.graph.latency_mrr import render_figure6
from script.artifact.graph.latency_breakdown import render_figure7
from script.artifact.graph.communication import pretty_print_comm
from script.artifact.graph.memory_consumption import pretty_print_mem
from script.artifact.graph.ablation_study import render_figure8

def run_performance(verbose):
    s_instance = create_instance(server_config, "ae-perf-server")
    c_instance = create_instance(client_config, "ae-perf-client")

    # waiting a bit for server to be ready
    # print("Sleep a bit for instances to be ready.")
    check_ssh_instances([s_instance, c_instance])

    # accuracy
    run_compass_accuracy(s_instance, verbose)

    # latency
    run_compass_latency(s_instance, c_instance, verbose)

    # baseline
    run_baseline_latency(s_instance, c_instance, verbose)

    delete_instance(PROJECT_ID, ZONE, s_instance["name"])
    delete_instance(PROJECT_ID, ZONE, c_instance["name"])
    return 


def run_ablation(verbose):
    s_instance = create_instance(server_config, "ae-ablation-server")
    c_instance = create_instance(client_config, "ae-ablation-client")

    check_ssh_instances([s_instance, c_instance])

    # c_instance = {
    #     "name": "ae-client",
    #     "internal_ip": "10.128.0.43"
    # }

    # s_instance = {
    #     "name": "ae-server",
    #     "internal_ip": "10.128.0.42"
    # }

    # accuracy 
    # run_ablation_accuracy(s_instance, verbose)
    run_ablation_accuracy_optimized(s_instance, verbose)

    # latency
    run_ablation_latency(s_instance, c_instance, verbose)

    delete_instance(PROJECT_ID, ZONE, s_instance["name"])
    delete_instance(PROJECT_ID, ZONE, c_instance["name"])

def run_tp(verbose):
    s_instance = create_instance(server_config, "ae-server-2")
    tracers = create_tracers(client_config, 25)

    check_ssh_instances([s_instance] + tracers)

    run_throughput(s_instance, tracers, verbose)

    tracer_names = [tracer["name"] for tracer in tracers]
    delete_tracers(PROJECT_ID, ZONE, tracer_names)
    delete_instance(PROJECT_ID, ZONE, s_instance["name"])


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Script for artifact evaluation")
        
        parser.add_argument(
            "--task",
            choices=["performance", "ablation", "throughput","stop_instances"],
            help="specify the task to run: performance, ablation, or stop all ae instances"
        )

        parser.add_argument(
            "--plot",
            choices=["figure6", "figure7", "table3", "table4", "figure8"],
            help="specify the graph / table to render: figure6, figure7, table3, table4, figure8"
        )

        parser.add_argument(
            "--verbose", 
            action="store_true",
            help="enable verbose output")

        args = parser.parse_args()

        if (args.task is None) == (args.plot is None):
            parser.error('Exactly one of --task or --plot must be specified.')

        if args.task:
            if args.task == "performance":
                print("Running performance exp")
                run_performance(args.verbose)
            elif args.task == "ablation":
                print("Running ablation exp")
                run_ablation(args.verbose)
            elif args.task == "throughput":
                print("Running thoughput exp")
                run_tp(args.verbose)
            elif args.task == "stop_instances":
                stop_instances()
            
        if args.plot:
            if args.plot == "figure6":
                render_figure6()
            elif args.plot == "figure7":
                render_figure7()
            elif args.plot == "table3":
                pretty_print_comm()
            elif args.plot == "table4":
                pretty_print_mem()
            elif args.plot == "figure8":
                render_figure8()

        
    except subprocess.CalledProcessError as e:
        print("The command")
        cmd_str = " ".join("'" + a + "'" if " " in a else a for a in e.cmd)
        print(f"\t{cmd_str}")
        print(f"failed with exit code {e.returncode}")
        sys.exit(e.returncode)