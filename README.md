# Generative AI Medical Application Backend

## Introduction
In the age of information, the medical field is constantly evolving. With the rise of holistic and integrative approaches to health, there's a growing need to cross-reference conventional medical knowledge with natural and herbal therapies. The HealthUniverse Hackathon presented an opportunity to address this need, leading to the development of our Generative AI Medical Application.

This repo covers two approaches:
* Meta Llama2 LLM model deployed on <a href="https://github.com/rmadabusiml/mediverse-api/blob/main/api/app/api/api_v1/endpoints/initialize.py#L117" target="_blank">SageMaker endpoint</a> via SageMaker Jumpstart 
* Anthropic Claude V2 model deployed on <a href="https://github.com/rmadabusiml/mediverse-api/blob/main/api/app/api/api_v1/endpoints/initialize.py#L131" target="_blank">AWS Bedrock endpoint</a>

## High-level Design

<p align="center" width="100%">
<img src="assets/hu_design.png" style="width: 70%; min-width: 300px; display: block; margin: auto;">
</p>

## Architecture with SageMaker JumpStart

<p align="center" width="100%">
<img src="assets/hu_architecture.png" style="width: 70%; min-width: 300px; display: block; margin: auto;">
</p>

## Architecture with AWS Bedrock

<p align="center" width="100%">
<img src="assets/hu_architecture_bedrock.png" style="width: 70%; min-width: 300px; display: block; margin: auto;">
</p>

## Demo Video

<p align="left" width="100%">
<a href="https://www.infoservices.com/hackathon/health-universe/mediverse_full_walkthru.mp4" target="_blank"><img src="assets/hu_demo.png" style="width: 70%; min-width: 300px; display: block; margin: auto;"></a>
</p>
