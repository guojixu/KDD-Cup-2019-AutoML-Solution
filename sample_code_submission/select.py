

# from util import log, timeit
# from FeatureSelector import FeatureSelector
#
# @timeit
# def feature_selector(X, y):
#     fs = FeatureSelector(data=X, labels=y)
#     fs.identify_single_unique()
#     print(fs.ops)
#
#     X = fs.remove(methods=['single_unique'], keep_one_hot=False)
#     cols_selected = X.columns.tolist()
#     return X, cols_selected
