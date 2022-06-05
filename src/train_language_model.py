if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForPreTraining, AutoModel, get_scheduler
    from torch.utils.data import DataLoader
    from torch.optim import AdamW

    from datasets.articles_dataset import ArticlesDataset

    DATA_DIR = "../data/"

    model = AutoModelForPreTraining.from_pretrained('neuralmind/bert-large-portuguese-cased')
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased', do_lower_case=True)

    article_train_idx = []
    article_test_idx = []
    with open("f{DATA_DIR}/articles_train.txt", "r") as f:
        article_train_idx = [int(line.strip("\n")) for line in f.readlines()]

    with open("f{DATA_DIR}/articles_test.txt", "r") as f:
        article_test_idx = [int(line.strip("\n")) for line in f.readlines()]

    training_data = ArticlesDataset(
        article_excel="OpArticles.xlsx",
        slice=article_train_idx,
        data_directory=DATA_DIR,
        use_transform=True,
        transform_p=0.5,
        use_synonyms=True,
        synonyms_json="synonyms.json"
    )

    testing_data = ArticlesDataset(
        article_excel="OpArticles.xlsx",
        slice=article_test_idx,
        data_directory=DATA_DIR,
        use_transform=False,
        use_synonyms=False,
    )

    # PARAMETERS
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    NUM_EPOCHS = 5

    train_dataloader = DataLoader(
        dataset=training_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True
    )

    test_dataloader = DataLoader(
        dataset=testing_data,
        batch_size=1,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False
    )

    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_training_steps = NUM_EPOCHS * len(train_dataloader)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

