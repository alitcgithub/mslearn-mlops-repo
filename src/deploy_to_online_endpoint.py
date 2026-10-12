import argparse
import datetime
from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import (
    DataCollector,
    DeploymentCollection,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    Model,
)
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential


def get_data_collector() -> DataCollector:
    return DataCollector(
        collections={
            "model_inputs": DeploymentCollection(enabled="true"),
            "model_outputs": DeploymentCollection(enabled="true"),
        }
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--subscription-id", dest="subscription_id", required=True)
    parser.add_argument("--resource-group", dest="resource_group", required=True)
    parser.add_argument("--workspace", dest="workspace", required=True)
    parser.add_argument(
        "--endpoint-name", dest="endpoint_name", default="diabetes-endpoint"
    )
    parser.add_argument(
        "--deployment-name", dest="deployment_name", default="green"
    )

    return parser.parse_args()


def get_ml_client(
    subscription_id: str, resource_group: str, workspace: str
) -> MLClient:
    credential = DefaultAzureCredential()
    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )


def ensure_endpoint(ml_client: MLClient, endpoint_name: str) -> ManagedOnlineEndpoint:
    try:
        return ml_client.online_endpoints.get(name=endpoint_name)
    except ResourceNotFoundError:
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name,
            description="Online endpoint for MLflow diabetes model",
            auth_mode="key",
        )
        return ml_client.online_endpoints.begin_create_or_update(
            endpoint
        ).result()


def register_model(ml_client: MLClient, model_name: str = "diabetes-model") -> Model:
    """Registers the model explicitly in the Azure ML Workspace Model Registry."""
    print(f"Registering model asset '{model_name}' in workspace registry...")
    version_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model_entity = Model(
        name=model_name,
        path="./model",
        type=AssetTypes.MLFLOW_MODEL,
        description="MLflow diabetes classification model",
        version=version_str,
    )

    registered_model = ml_client.models.create_or_update(model_entity)
    print(
        f"Model registered successfully: {registered_model.name} (Version: {registered_model.version})"
    )
    return registered_model


def create_or_update_deployment(
    ml_client: MLClient,
    endpoint_name: str,
    deployment_name: str,
    model: Model,
) -> ManagedOnlineDeployment:
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,
        instance_type="Standard_D2as_v4",
        instance_count=1,
        data_collector=get_data_collector(),
    )

    return ml_client.online_deployments.begin_create_or_update(
        deployment
    ).result()


def set_traffic_to_deployment(
    ml_client: MLClient, endpoint_name: str, deployment_name: str
) -> None:
    endpoint = ml_client.online_endpoints.get(name=endpoint_name)
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).result()


def main() -> None:
    args = parse_args()

    print("Connecting to Azure Machine Learning workspace...")
    ml_client = get_ml_client(
        subscription_id=args.subscription_id,
        resource_group=args.resource_group,
        workspace=args.workspace,
    )

    print(f"Ensuring online endpoint '{args.endpoint_name}' exists...")
    endpoint = ensure_endpoint(ml_client, args.endpoint_name)
    print(f"Using endpoint: {endpoint.name}")

    # Register model explicitly before creating/updating deployment
    registered_model = register_model(ml_client, model_name="diabetes-model")

    print(f"Creating or updating deployment '{args.deployment_name}'...")
    deployment = create_or_update_deployment(
        ml_client=ml_client,
        endpoint_name=endpoint.name,
        deployment_name=args.deployment_name,
        model=registered_model,
    )
    print(f"Deployment state: {deployment.provisioning_state}")

    print("Directing 100% of traffic to the deployment...")
    set_traffic_to_deployment(ml_client, endpoint.name, args.deployment_name)

    endpoint = ml_client.online_endpoints.get(name=endpoint.name)
    print(f"Deployment complete. Scoring URI: {endpoint.scoring_uri}")


if __name__ == "__main__":
    main()
