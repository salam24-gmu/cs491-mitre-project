import filter2
class analize_sentiment:
    def run_sent(filename, id, val):
        f = filter2.filter2()
        tweets = f.preprocess(filename, id, val)
        for user in tweets:
            print(user)




analize_sentiment.run_sent("mock_datasets/data_gen_example.csv", 0, 1)

    