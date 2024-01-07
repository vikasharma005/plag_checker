# main.py
import utility
import model
import pandas as pd

from clean import clean

def main():
    # Fetching Data
    df_train = pd.read_csv('data/train.csv', encoding='utf-8')
    df_train['id'] = df_train['id'].apply(str)

    df_test = pd.read_csv('data/test.csv', encoding='utf-8')
    df_test['test_id'] = df_test['test_id'].apply(str)

    df_all = pd.concat((df_train, df_test))
    df_all['question1'].fillna('', inplace=True)
    df_all['question2'].fillna('', inplace=True)

    q1_train = df_train['question1'][:10000]
    q2_train = df_train['question2'][:10000]
    label = df_train['is_duplicate'][:10000]

    # Cleaning Data
    for i in range(len(q1_train)):
        q1_train[i] = clean(q1_train[i])

    for i in range(len(q2_train)):
        q2_train[i] = clean(q2_train[i])

    # Processing Train Data
    X_train_l, X_train_r, embedding_matrix, word_to_index = utility.process_train(q1_train, q2_train)

    # Processing Test Data
    emb_q1, emb_q2 = utility.process_test(df_train.question1[1001],
                                          df_train.question2[1001],
                                          word_to_index,
                                          dimx=50, dimy=50)

    # Training Model
    trained_model = model.train_model(X_train_l, X_train_r, label, embedding_matrix)

    # Making Predictions
    predictions = utility.predict(trained_model, emb_q1, emb_q2)
    print("Plagiarism Prediction:", predictions)

if __name__ == "__main__":
    main()
