import os
# import yaml
import numpy as np
from fastapi import FastAPI
from schema import ModelInput
from pandas import DataFrame
from joblib import load
# from source.train_model import infer
# from source.train_model import process_data, cat_features




if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    os.system("dvc remote add -df s3-bucket s3://udacitycoursebucket")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


app = FastAPI()


@app.get("/")
async def get_items():
    return {"message": "Welcome!"}


# @app.post("/")
# async def inference(input_data: ModelInput):
#     model = load("model/model.joblib")
#     encoder = load("model/encoder.joblib")
#     lb = load("model/lb.joblib")
#     array = np.array([[
#                      input_data.age,
#                      input_data.workclass,
#                      input_data.education,
#                      input_data.maritalStatus,
#                      input_data.occupation,
#                      input_data.relationship,
#                      input_data.race,
#                      input_data.sex,
#                      input_data.hoursPerWeek,
#                      input_data.nativeCountry
#                      ]])
#     df_temp = DataFrame(data=array, columns=[
#         "age",
#         "workclass",
#         "education",
#         "marital-status",
#         "occupation",
#         "relationship",
#         "race",
#         "sex",
#         "hours-per-week",
#         "native-country",
#     ])
#     X, _, _, _ = process_data(
#                 df_temp,
#                 categorical_features=cat_features,
#                 encoder=encoder, lb=lb, training=False)
#     prediction = infer(model, X)
#     y = lb.inverse_transform(prediction)[0]

#     return {"prediction": y}