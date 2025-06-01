https://tradebotbtc-api-461917555020.asia-southeast2.run.app/health

https://tradebotbtc-api-461917555020.asia-southeast2.run.app/model-status

docker built -t tradebott-btc .

install gcp
(New-Object Net.WebClient).DownloadFile("https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe", "$env:Temp\GoogleCloudSDKInstaller.exe")

& $env:Temp\GoogleCloudSDKInstaller.exe
    

containerregistry.googleapis.com
gcloud services enable containerregistry.googleapis.com

gcloud config set project tradebotbtc
docker tag tradebott-btc gcr.io/my-gcp-project/tradebott-btc
gcloud auth configure-docker
docker push gcr.io/my-gcp-project/tradebott-btc
gcloud services enable run.googleapis.com
gcloud run deploy tradebotbtc-api \
gcloud run deploy tradebotbtc-api --image gcr.io/my-gcp-project/tradebott-btc --platform managed --region asia-southeast2 --allow-unauthenticated


# 1. Set the correct project
gcloud config set project tradebotbtc

# 2. Tag the local Docker image for GCP
docker tag tradebott-btc gcr.io/tradebotbtc/tradebott-btc

# 3. Authenticate Docker to push to GCP
gcloud auth configure-docker

# 4. Push image to GCP Container Registry
docker push gcr.io/tradebotbtc/tradebott-btc

# 5. Enable Cloud Run service
gcloud services enable run.googleapis.com

# 6. Deploy to Cloud Run
gcloud run deploy tradebotbtc-api --image gcr.io/tradebotbtc/tradebott-btc --platform managed --region asia-southeast2 --allow-unauthenticated


### DEPLOY ONLY 
# 2. Tag the local Docker image for GCP
docker tag tradebott-btc gcr.io/tradebotbtc/tradebott-btc

# 4. Push image to GCP Container Registry
docker push gcr.io/tradebotbtc/tradebott-btc

# 6. Deploy to Cloud Run
gcloud run deploy tradebotbtc-api --image gcr.io/tradebotbtc/tradebott-btc --platform managed --region asia-southeast2 --allow-unauthenticated



Adding credentials for all GCR repositories.
WARNING: A long list of credential helpers may cause delays running 'docker build'. We recommend passing the registry name to configure only the registry you are using.
After update, the following will be written to your Docker config file located at [C:\Users\ASUS
ROG\.docker\config.json]:
 {
  "credHelpers": {
    "gcr.io": "gcloud",
    "us.gcr.io": "gcloud",
    "eu.gcr.io": "gcloud",
    "asia.gcr.io": "gcloud",
    "staging-k8s.gcr.io": "gcloud",
    "marketplace.gcr.io": "gcloud"
  }
}

Do you want to continue (Y/n)?
