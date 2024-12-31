from training_pipeline import training_pipeline
from prefect import flow
# from prefect import PREFECT_DEFAULT_WORK_POOL_NAME

@flow(name="Deployment_Pipeline", log_prints=True)
def deployment_pipeline():
    # Deploy the pipeline
    deployment_id = training_pipeline.deploy(
        name="Deployment_Pipeline", 
        work_pool_name="deployment_workpool",
        image=None, # Docker image   
        interval=60, 
        tags=["onboarding"],  
        parameters={"Success":True},
        description="Deployment for training pipeline",
        paused=True,  
    )
    print(f"Deployment created successfully with ID: {deployment_id}")

    # Optionally stop after deployment
    print("Deployment completed. Stopping the script.")
    exit(0)

if __name__ == "__main__":
    deployment_pipeline()
