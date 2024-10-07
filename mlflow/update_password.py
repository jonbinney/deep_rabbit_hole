#!/bin/env python3
# Simple python script to update the default MLFlow admin password
import requests
from getpass import getpass

host = "https://soft-pond-5082.ploomberapp.io"
password_new = "W3AreLazy!"

response = requests.patch(
    f"{host}/api/2.0/mlflow/users/update-password",
    auth=("admin", "password"),
    json={"username": "admin", "password": password_new},
)

response.raise_for_status()

print("Your new password is:", password_new)