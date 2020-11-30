---
layout: post
title: Setting up Prefect
---

### Installing prefect:
```
pip install prefect
```

### Setup for connecting to prefect UI:
Change current working directory to the following path:  

```
cd ~/.prefect
```

Paste the config.toml from repository in working directory.
Go to the following url : http://192.168.1.19:7888/

Ensure you are connected (refer right top of below image). 
If not enter http://192.168.1.19:7887/graphql in the text box highlighted in the image below and click connect.


![landing_page](https://bitbucket.org/aiinnovation/prefect-pipelines/downloads/prefect_landing_page_setup.PNG)

Incase you need to start a prefect server run the following command after pasting config.toml to ./prefect
```
prefect backend server
prefect server start
```
In a new window enter the following:
```
prefect backend server
prefect server create-tenant --name Qureai --slug qureai
```