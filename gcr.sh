#!/bin/bash

gcloud builds submit --tag eu.gcr.io/flags-293518/image1 .

gcloud container clusters create flags \
--num-nodes 1 \
--enable-basic-auth \
--issue-client-certificate \
--zone europe-west3-a

kubectl get nodes

kubectl apply -f deployment.yaml

kubectl get deployments

kubectl apply -f service.yaml

kubectl get services