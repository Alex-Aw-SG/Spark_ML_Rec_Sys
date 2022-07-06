# Spark ML Recommender System

Project Objective:

Develop big data recommender system using Spark ML on AWS EMR cluster.
Repo consist of scripts to run Spark ML.

Steps and process:
1) First notebook sandbox codes tested on Google colab.
2) PY file was converted from notebook which was used on AWS EMR.
   Script was part of the Data Pipeline that would automatically train 1.5M
   data of beer reviews and generate Top 10 recommendations for each users.
3) Data was to be pushed to AWS S3 for storage which also triggers Lambda 
   to automatically save the information in DynamoDB.
4) DynamoDB was used to for data retrieval directly from web via Gateway.

![architecture](https://github.com/Alex-Aw-SG/Spark_ML_Rec_Sys/blob/master/Image/architecture.jpg)

