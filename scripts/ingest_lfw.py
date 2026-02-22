from sklearn.datasets import fetch_lfw_people

def ingest_lfw_dataset():
    lfw_people = fetch_lfw_people(data_home="./data")
    print(lfw_people.data.shape)
    print(lfw_people.target.shape)
    return lfw_people