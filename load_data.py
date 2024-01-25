import pandas

questions = pandas.read_csv("./questions.csv", encoding="ISO-8859-1")
#tags = pandas.read_csv("./tags.csv", encoding="ISO-8859-1")


tags = pandas.read_csv("./tags.csv", na_filter=False, encoding="ISO-8859-1")
# original str. statt NAN für null, NA