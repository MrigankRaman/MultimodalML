from models import *
from datasets import *
from tqdm import tqdm   
import ipdb
import csv
from sklearn.metrics import precision_recall_fscore_support

def test(args):
    model = ModelWrapper(args.model)
    #check if image_size exists in model.config
    # print(model.model.config.num_images)
    if hasattr(model.model.config, 'image_size'):
        image_size = model.model.config.image_size
    else:
        image_size = None
    eval_data = NLVR2Dataset('/home/mrigankr/PGM/nlvr2/data/dev.json', args.model, image_size = image_size, image_path="/data/mrigankr/mml/dev/")
    if args.path != "None":
        model.load_state_dict(torch.load(f"{args.model_dir}/{args.path}"))
    model.eval()
    model.model.cuda()
    eval_dataloader = DataLoader(eval_data, batch_size=args.test_batch_size, shuffle=False, num_workers=4)
    preds = []
    labels = []
    for batch in tqdm(eval_dataloader):
        with torch.no_grad():
            # ipdb.set_trace()
            outputs = model(**batch)
            logits = outputs[1]
            preds.append(logits)
            labels.append(batch['labels'])
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    predictions = np.argmax(preds.cpu().detach().numpy() , axis=-1)
    labels = labels.cpu().detach().numpy()

    # Write results to file
    identifiers = [i['identifier'] for i in eval_data.examples]
    results = zip(identifiers, predictions, labels)
    with open(f"{args.model_dir}/results.csv", "w") as f:
        writer = csv.writer(f)
        # writer.writerow(["id", "prediction", "label"])
        writer.writerows(results)
        

    print("Accuracy: ",(predictions == labels).astype(np.float32).mean().item())
    metrics = precision_recall_fscore_support(predictions, labels, average='binary')
    precision, recall, f1, _ = metrics
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)