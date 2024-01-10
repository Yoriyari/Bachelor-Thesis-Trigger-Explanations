"""
    Fast API Endpoint file

    Designed to be "dumb", does some data validation and data transformation, but really passes off
    the heavy lifting further down the pipeline.

    To run this model-training api, run the following command (assuming all packages are installed):
        `uvicorn main:app --reload --port=9000`

    To access interactive docs and dummy test data navigate to http://127.0.0.1:9000/docs after running the
    above command
"""
import json
import os
import sys
import torch
from fastapi import FastAPI, BackgroundTasks
from fastapi import status

sys.path.append(".")
sys.path.append("../model_training/")
import fast_api_util_functions as util_f
import json_schema as schema
from internal_api.internal_main import train_next_framework_lean_life, train_next_framework, apply_strict_matching, \
    apply_soft_matching, evaluate_next, train_standard_pipeline, \
    train_standard_lean_life, evaluate_standard, predict_next, predict_standard, train_standard_ner_pipeline, \
    train_standard_ner_lean_life, evaluate_standard_ner, predict_standard_ner, train_trigger_soft_match_pipeline, \
    evaluate_trigger_ner, predict_trigger_ner, train_trigger_soft_match_lean_life

# We don't have a sophisticated CUDA Management policy, so please make needed changes to fit your needs
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
app = FastAPI()


@app.post("/training/next/lean-life/", status_code=status.HTTP_201_CREATED)
async def start_next_training_lean_life(lean_life_payload: schema.LeanLifePayload, background_tasks: BackgroundTasks):
    """
        Endpoint hit by annotation tool's django api
    """
    params = lean_life_payload.params
    lean_life_data = lean_life_payload.lean_life_data
    prepped_data = util_f.prepare_next_data(lean_life_data, project_type=params.project_type)
    label_space, unlabeled_docs, explanation_triples, ner_label_space = prepped_data
    if len(ner_label_space) == 0:
        ner_label_space = None
    if len(unlabeled_docs) == 0:
        unlabeled_docs = None
    if len(explanation_triples) == 0:
        explanation_triples = None
    background_tasks.add_task(train_next_framework_lean_life, params.__dict__, label_space, unlabeled_docs,
                              explanation_triples, ner_label_space)


@app.post("/training/next/api/", status_code=status.HTTP_200_OK, response_model=schema.SavePathOutput)
async def start_next_training_api(api_payload: schema.ExplanationTrainingPayload):
    """
        Endpoint used to kick off training of a classifier via the next framework of learning from
        explanations. Please refer to the docs or `json_schema.py` to understand both the supported
        and required paramaters.
    """
    params = api_payload.params
    prepped_data = util_f.prepare_next_data(api_payload, lean_life=False)
    label_space, unlabeled_docs, explanation_triples, ner_label_space = prepped_data
    data = params.dict()
    if len(ner_label_space) == 0:
        ner_label_space = None
    if len(unlabeled_docs) == 0:
        unlabeled_docs = None
    if len(explanation_triples) == 0:
        explanation_triples = None
    if hasattr(api_payload, "dev_data") and api_payload.dev_data is not None:
        data["dev_data"] = api_payload.dev_data
        for i, doc in enumerate(data["dev_data"]):
            data["dev_data"][i] = doc.dict()
    else:
        data["dev_data"] = None
    return schema.SavePathOutput(save_path=train_next_framework(data, label_space, unlabeled_docs,
                                                                explanation_triples, ner_label_space))


@app.post("/training/next/eval", status_code=status.HTTP_200_OK, response_model=schema.NextEvalDataOutput)
async def eval_next_clf(api_payload: schema.EvalNextClfPayload):
    """
        Endpoint used to evaluate a classifier trained via the next framework. Please refer to the docs or
        `json_schema.py` to understand both the supported and required paramaters.
    """
    params = api_payload.params.dict()
    params["label_map"] = api_payload.label_space
    params["eval_data"] = api_payload.eval_data
    ner_label_space = api_payload.ner_label_space
    avg_loss, avg_eval_ent_f1_score, avg_eval_val_f1_score, no_relation_thresholds = evaluate_next(params, ner_label_space)
    return schema.NextEvalDataOutput(
        avg_loss=avg_loss, avg_eval_ent_f1_score=avg_eval_ent_f1_score,
        avg_eval_val_f1_score=avg_eval_val_f1_score,
        no_relation_thresholds=no_relation_thresholds)


@app.post("/training/next/predict", status_code=status.HTTP_200_OK, response_model=schema.PredictionOutputs)
async def predict_next_clf(api_payload: schema.PredictNextApiParams):
    """
        Endpoint used to predict from a classifier trained via the next framework. Please refer to the docs or
        `json_schema.py` to understand both the supported and required parameters.
    """
    params = api_payload.params.dict()
    params["label_map"] = api_payload.label_space
    params["prediction_data"] = api_payload.prediction_data
    ner_label_space = api_payload.ner_label_space
    probs, preds = predict_next(params, ner_label_space)
    return schema.PredictionOutputs(class_probs=probs, class_preds=preds)


@app.get("/download/{file_path:path}")
async def get_trained_model(file_path: str):
    """
        Endpoint used to load a saved model's weight and send them back to requester

        Model weights are the model's state_dict, but instead of tensors we save the weights
        in Python Lists so that we can serialize the state_dict. To get back the original
        state_dict, per key convert the list back to a PyTorch Tensor.
    """
    state_dict = torch.load(file_path, map_location="cpu")
    for key in state_dict:
        state_dict[key] = state_dict[key].numpy().tolist()
    return json.loads(json.dumps(state_dict))


@app.post("/other/next/match", status_code=status.HTTP_200_OK, response_model=schema.MatchedDataOutput)
async def strict_match_data(api_payload: schema.StrictMatchPayload):
    """
        Endpoint that converts explanations into strict labeling functions and labels
        a pool of unlabeled sentences. Please refer to the docs or `json_schema.py` to
        understand the required paramaters.
    """
    api_payload = api_payload.dict()
    result = apply_strict_matching(api_payload)
    return schema.MatchedDataOutput.parse_obj({"matched_tuples": result[0], "matched_indices": result[1]})


@app.post("/other/next/softmatch", status_code=status.HTTP_200_OK, response_model=schema.SoftMatchData)
async def soft_match_data(api_payload: schema.LeanLifePayload):
    """
        Endpoint that converts explanations into strict labeling functions and labels
        a pool of unlabeled sentences. Please refer to the docs or `json_schema.py` to
        understand the required paramaters.
    """
    params = api_payload.params
    lean_life_data = api_payload.lean_life_data
    prepped_data = util_f.prepare_next_data(lean_life_data, project_type=params.project_type)
    label_space, unlabeled_docs, explanation_triples, ner_label_space = prepped_data
    if len(ner_label_space) == 0:
        ner_label_space = None
    if len(unlabeled_docs) == 0:
        unlabeled_docs = None
    if len(explanation_triples) == 0:
        explanation_triples = None
    data = apply_soft_matching(params.__dict__, label_space, unlabeled_docs, explanation_triples,
                               ner_label_space)
    return schema.SoftMatchData(scores=data)


@app.post("/training/standard/api/", status_code=status.HTTP_200_OK, response_model=schema.SavePathOutput)
async def start_standard_training_api(api_payload: schema.StandardPipelinePayload):
    """
        Endpoint used to kick off training of a classifier.
        Please refer to the docs or `json_schema.py` to understand both the supported and required parameters.
    """
    params = api_payload.params
    prepped_data = util_f.prepare_standard_data(api_payload, lean_life=False)
    label_space, labeled_docs, dev_docs, ner_label_space = prepped_data
    if len(ner_label_space) == 0:
        ner_label_space = None
    if len(labeled_docs) == 0:
        labeled_docs = None
    if len(dev_docs) == 0:
        dev_docs = None
    return schema.SavePathOutput(
        save_path=train_standard_pipeline(params.dict(), label_space, labeled_docs, dev_docs, ner_label_space)
    )


@app.post("/training/standard/lean-life/", status_code=status.HTTP_201_CREATED)
async def start_standard_training_lean_life(lean_life_payload: schema.LeanLifeStandardPayload,
                                            background_tasks: BackgroundTasks):
    """
        Endpoint hit by annotation tool's django api
    """
    params = lean_life_payload.params
    lean_life_data = lean_life_payload.lean_life_data
    prepped_data = util_f.prepare_standard_data(lean_life_data, project_type=params.project_type)
    label_space, labeled_docs, dev_docs, ner_label_space = prepped_data
    if len(ner_label_space) == 0:
        ner_label_space = None
    if len(labeled_docs) == 0:
        labeled_docs = None
    if len(dev_docs) == 0:
        dev_docs = None
    background_tasks.add_task(train_standard_lean_life, params.dict(), label_space, labeled_docs, dev_docs,
                              ner_label_space)


@app.post("/training/standard/eval", status_code=status.HTTP_200_OK, response_model=schema.StandardEvalDataOutput)
async def eval_standard_clf(api_payload: schema.EvalStandardClfPayload):
    """
        Endpoint used to evaluate a classifier trained via the standard pipeline. Please refer to the docs or
        `json_schema.py` to understand both the supported and required parameters.
    """
    params = api_payload.params.dict()
    params["label_map"] = api_payload.label_space
    params["eval_data"] = api_payload.eval_data
    ner_label_space = api_payload.ner_label_space
    if ner_label_space is not None and len(ner_label_space) == 0:
        ner_label_space = None
    avg_loss, avg_eval_f1_score = evaluate_standard(params, ner_label_space)
    return schema.StandardEvalDataOutput(avg_loss=avg_loss, avg_eval_f1_score=avg_eval_f1_score)


@app.post("/training/standard/predict", status_code=status.HTTP_200_OK, response_model=schema.PredictionOutputs)
async def predict_standard_clf(api_payload: schema.PredictStandardClfPayload):
    """
        Endpoint used to predict from a classifier trained via the standard pipeline. Please refer to the docs or
        `json_schema.py` to understand both the supported and required parameters.
    """
    params = api_payload.params.dict()
    params["label_map"] = api_payload.label_space
    params["prediction_data"] = api_payload.prediction_data
    ner_label_space = api_payload.ner_label_space
    if ner_label_space is not None and len(ner_label_space) == 0:
        ner_label_space = None
    probs, preds = predict_standard(params, ner_label_space)
    return schema.PredictionOutputs(class_probs=probs, class_preds=preds)


@app.post("/training/standard/ner/lean-life/", status_code=status.HTTP_201_CREATED)
async def start_standard_training_ner_lean_life(lean_life_payload: schema.LeanLifeStandardNERPayload,
                                            background_tasks: BackgroundTasks):
    """
        Endpoint hit by annotation tool's django api
    """
    params = lean_life_payload.params
    lean_life_data = lean_life_payload.lean_life_data
    prepped_data = util_f.prepare_standard_data(lean_life_data, project_type=params.project_type)
    _, labeled_docs, dev_docs, _ = prepped_data
    if len(labeled_docs) == 0:
        labeled_docs = None
    if len(dev_docs) == 0:
        dev_docs = None
    background_tasks.add_task(train_standard_ner_lean_life, params.dict(), labeled_docs, dev_docs)


@app.post("/training/standard/ner/api/", status_code=status.HTTP_200_OK, response_model=schema.SavePathOutput)
async def start_standard_training_ner_api(api_payload: schema.StandardNERTrainingPayload):
    """
        Endpoint used to kick off training of a classifier.
        Please refer to the docs or `json_schema.py` to understand both the supported and required parameters.
    """
    params = api_payload.params
    prepped_data = util_f.prepare_standard_data(api_payload, lean_life=False)
    _, labeled_docs, dev_docs, _ = prepped_data
    if len(labeled_docs) == 0:
        labeled_docs = None
    if len(dev_docs) == 0:
        dev_docs = None
    if hasattr(api_payload, "eval_data") and api_payload.dev_data is not None:
        eval_docs = api_payload.eval_data
        for i, doc in enumerate(eval_docs):
            eval_docs[i] = doc.dict()
    else:
        eval_docs = None
    return schema.SavePathOutput(
                    save_path=train_standard_ner_pipeline(params.dict(), labeled_docs, dev_docs, eval_docs))


@app.post("/training/standard/ner/eval", status_code=status.HTTP_200_OK, response_model=schema.StandardNEREvalDataOutput)
async def start_eval_standard_ner(api_payload: schema.EvalStandardNERPayload):
    """
        Endpoint used to evaluate a classifier trained via the standard pipeline. Please refer to the docs or
        `json_schema.py` to understand both the supported and required parameters.
    """
    params = api_payload.params.dict()
    params["eval_data"] = api_payload.eval_data
    precision, recall, f1 = evaluate_standard_ner(params)
    return schema.StandardNEREvalDataOutput(precision=precision, recall=recall, f1=f1)


@app.post("/training/standard/ner/predict", status_code=status.HTTP_200_OK, response_model=schema.StandardNERPredictionOutputs)
async def start_predict_standard_ner(api_payload: schema.PredictStandardNERPayload):
    """
        Endpoint used to predict from a classifier trained via the standard pipeline. Please refer to the docs or
        `json_schema.py` to understand both the supported and required parameters.
    """
    params = api_payload.params.dict()
    params["prediction_data"] = api_payload.prediction_data
    preds = predict_standard_ner(params)
    return schema.StandardNERPredictionOutputs(class_preds=preds)


@app.post("/training/trigger/lean-life/", status_code=status.HTTP_201_CREATED)
async def start_trigger_training_lean_life(lean_life_payload: schema.LeanLifeTriggerPayload,
                                           background_tasks: BackgroundTasks):
    """
        Endpoint hit by annotation tool's django api
    """
    params = lean_life_payload.params
    lean_life_data = lean_life_payload.lean_life_data
    prepped_data = util_f.prepare_next_data(lean_life_data, project_type=params.project_type)
    _, unlabeled_docs, explanation_triples, _ = prepped_data
    if len(unlabeled_docs) == 0:
        unlabeled_docs = None
    if len(explanation_triples) == 0:
        explanation_triples = None
    background_tasks.add_task(train_trigger_soft_match_lean_life, params.__dict__,unlabeled_docs,
                              explanation_triples)


@app.post("/training/trigger/api/", status_code=status.HTTP_200_OK, response_model=schema.SavePathOutput)
async def start_trigger_training_api(api_payload: schema.ExplanationTriggerTrainingPayload):
    """
        Endpoint used to kick off training of a classifier via the trigger framework of learning from
        explanations. Please refer to the docs or `json_schema.py` to understand both the supported
        and required parameters.
    """
    params = api_payload.params
    prepped_data = util_f.prepare_next_data(api_payload, lean_life=False)
    _, unlabeled_docs, explanation_triples, _ = prepped_data
    data = params.dict()
    if len(unlabeled_docs) == 0:
        unlabeled_docs = None
    if len(explanation_triples) == 0:
        explanation_triples = None

    if hasattr(api_payload, "dev_data") and api_payload.dev_data is not None:
        data["dev_data"] = api_payload.dev_data
        for i, doc in enumerate(data["dev_data"]):
            data["dev_data"][i] = doc.dict()
    else:
        data["dev_data"] = None

    if hasattr(api_payload, "eval_data") and api_payload.dev_data is not None:
        eval_docs = api_payload.eval_data
        for i, doc in enumerate(eval_docs):
            eval_docs[i] = doc.dict()
    else:
        eval_docs = []

    data["eval_data"] = eval_docs
    return schema.SavePathOutput(save_path=train_trigger_soft_match_pipeline(
                                    data, unlabeled_docs, explanation_triples
                                ))


@app.post("/training/trigger/eval", status_code=status.HTTP_200_OK, response_model=schema.StandardNEREvalDataOutput)
async def eval_trigger(api_payload: schema.EvalStandardNERPayload):
    """
        Endpoint used to evaluate a classifier trained via the trigger framework. Please refer to the docs or
        `json_schema.py` to understand both the supported and required paramaters.
    """
    params = api_payload.params.dict()
    params["eval_data"] = api_payload.eval_data
    precision, recall, f1 = evaluate_trigger_ner(params)
    return schema.StandardNEREvalDataOutput(precision=precision, recall=recall, f1=f1)


@app.post("/training/trigger/predict", status_code=status.HTTP_200_OK, response_model=schema.NERPredictionOutputs)
async def predict_trigger(api_payload: schema.PredictStandardNERPayload):
    """
        Endpoint used to predict from a classifier trained via the trigger framework. Please refer to the docs or
        `json_schema.py` to understand both the supported and required parameters.
    """
    params = api_payload.params.dict()
    params["prediction_data"] = api_payload.prediction_data
    preds, trigs, dists = predict_trigger_ner(params)
    return schema.NERPredictionOutputs(class_preds=preds, trigger_preds=trigs, distance_preds=dists)
