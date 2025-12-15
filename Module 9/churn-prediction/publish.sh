
ECR_URL=926630426627.dkr.ecr.eu-north-1.amazonaws.com
REPO_URL=${ECR_URL}/churn-prediction-lambda
LOCAL_IMAGE=churn-prediction-lambda

docker build --provenance=false -t ${LOCAL_IMAGE} .
# docker tag $LOCAL_IMAGE:latest $REPO_URL:latest
# docker push $REPO_URL:latest

aws ecr get-login-password \
  --region eu-north-1 \
| docker login \
  --username AWS \
  --password-stdin ${ECR_URL}

REMOTE_IMAGE_TAG="${ECR_URL}/churn-prediction-lambda:v1"

docker build --provenance=false -t churn-prediction-lambda .
docker tag churn-prediction-lambda ${REMOTE_IMAGE_TAG}
docker push ${REMOTE_IMAGE_TAG}
