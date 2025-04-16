def get_dataset_verbalizers(dataset: str):
    
    if dataset in ["sst2",  "mr", "cr"]:
        verbalizers = ["negative", "positive"]  # num_classes
    elif dataset == "agnews":
        verbalizers = ["World", "Sports", "Business", "Tech"]  # num_classes
    elif dataset in ["sst-5"]:
        verbalizers = [
            "terrible",
            "bad",
            "okay",
            "good",
            "great",
        ]  
    elif dataset == "trec":
        verbalizers = [
            "Description",
            "Entity",
            "Expression",
            "Human",
            "Location",
            "Number",
        ]
    return verbalizers