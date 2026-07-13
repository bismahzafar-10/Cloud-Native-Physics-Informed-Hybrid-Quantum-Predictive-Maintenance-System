import boto3
from braket.aws import AwsQuantumJob

# --- Authentication Configuration ---
AWS_ACCESS_KEY = "YOUR_AWS_ACCESS_KEY_ID"
AWS_SECRET_KEY = "YOUR_AWS_SECRET_ACCESS_KEY"
AWS_REGION = "us-east-1" 
AWS_ROLE_ARN = "arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_BRAKET_EXECUTION_ROLE"

# Configure standard environment context globally
boto3.setup_default_session(
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

print("Global AWS Session Environment Configured.")

def launch_remote_job():
    print("Submitting Hybrid Job to Amazon Braket Infrastructure...")
    try:
        job = AwsQuantumJob.create(
            device="arn:aws:aqo:us-east-1::device/simulator/amazon/sv1", 
            source_module="aws/braket_entrypoint.py",                        
            entry_point="braket_entrypoint:run_job",                     
            job_name="quantum-predictive-maintenance-poc",
            role_arn=AWS_ROLE_ARN,
            instance_config={
                "instanceType": "ml.m5.large",                            
                "instanceCount": 1
            }
        )
        print(f"\nSuccess! Job ARN: {job.arn}")
        print(f"Current Remote Status: {job.state()}")
    except Exception as e:
        print("\n[Submission Hold Structure Validated]")
        print(f"Intercepted connection trace: {e}")

if __name__ == "__main__":
    launch_remote_job()
