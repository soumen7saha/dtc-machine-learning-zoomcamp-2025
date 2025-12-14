import boto3
import json

lambda_client = boto3.client('lambda')

customer = ...

response = lambda_client.invoke(
    FunctionName='churn-prediction',
    InvocationType='RequestResponse',
    Payload=json.dumps(customer)
)

result = json.loads(response['Payload'].read())
print(json.dumps(result, indent=2))