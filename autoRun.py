import subprocess
import psutil
import time

def is_process_running(pid):
    """Check if a process with the given PID is currently running."""
    return psutil.pid_exists(pid)

def main():
    pid_to_monitor = 4123199  # Replace this with the PID of your running job
    secondary_script_path = "main.py"

    # Define arguments for the secondary script
    secondary_script_args = [
        "--max_epochs", "18",
        "--base", "configs/custom_vqgan.yaml",
        "-t", "True",
        "--gpus", "0,",
        "--resume", "/home/jqi/Github/taming-transformers/logs/2023-12-04T08-09-39_custom_vqgan"
    ]

    print(f"Monitoring process with PID: {pid_to_monitor}")

    # Continuously check if the process is still running
    while is_process_running(pid_to_monitor):
        print(f"Process {pid_to_monitor} is still running...")
        time.sleep(600)  # Wait for 5 seconds before checking again

    print(f"Process {pid_to_monitor} has finished.")

    # Process has finished, run the next script with arguments
    print("Running secondary script with arguments...")
    subprocess.run(["python", secondary_script_path] + secondary_script_args)

if __name__ == "__main__":
    main()
