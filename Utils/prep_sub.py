import pandas as pd
import numpy as np
import datetime
import csv
from tqdm import tqdm


def write_submission(recommender):
    targetUsers = pd.read_csv('../data/original_data/data_target_users_test.csv')['user_id']

    targetUsers = targetUsers.tolist()

    with open("../submissions/submission_" + datetime.datetime.now().strftime("%H_%M") + ".csv", 'w',
              newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user_id', 'item_list'])

        for userID in tqdm(targetUsers):
            writer.writerow([userID, str(np.array(recommender.recommend(userID, 10)))[1:-1]])

    print("Printing finished")


def write_submission_split_opt(recommenders: list, users: list):
    targetUsers = pd.read_csv('../data/original_data/data_target_users_test.csv')['user_id']

    targetUsers = targetUsers.tolist()

    with open("../submissions/submission_" + datetime.datetime.now().strftime("%H_%M") + ".csv", 'w',
              newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user_id', 'item_list'])

        for userID in tqdm(targetUsers):
            for i in range(len(users)):
                if userID in users[i]:
                    writer.writerow([userID, str(np.array(recommenders[i].recommend(userID, 10)))[1:-1]])
    print("Printing finished")
