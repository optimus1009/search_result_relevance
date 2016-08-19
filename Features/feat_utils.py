
"""
__file__

	feat_utils.py

__description__

	This file provides utils for generating features.

__author__

	xionglin <optimus1009@163.com>

"""

# 除法函数
def try_divide(x, y, val=0.0):
    """ 
    	Try to divide two numbers
    """
    if y != 0.0:
    	val = float(x) / y
    return val

# Group samples by median_relevance
def get_sample_indices_by_relevance(dfTrain, additional_key=None):
	""" 
		return a dict with
		key: (additional_key, median_relevance)
		val: list of sample indices
	"""
	# dfTrain.shape[0]  获取dfTrain的行数 range(dfTrain.shape(0)) = 0,1,2,.....,(dfTrain.shape[0]-1)
	dfTrain["sample_index"] = range(dfTrain.shape[0])
	group_key = ["median_relevance"]
	if additional_key != None:
		group_key.insert(0, additional_key)
	agg = dfTrain.groupby(group_key, as_index=False).apply(lambda x: list(x["sample_index"]))
	d = dict(agg)
	# DataFrame.drop(labels, axis=0, level=None, inplace=False, errors='raise')
	# Return new object with labels in requested axis removed.
	dfTrain = dfTrain.drop("sample_index", axis=1)
	return d


def dump_feat_name(feat_names, feat_name_file):
	"""
		save feat_names to feat_name_file
	"""
	with open(feat_name_file, "wb") as f:
	    for i,feat_name in enumerate(feat_names):
	        if feat_name.startswith("count") or feat_name.startswith("pos_of"):
	            f.write("('%s', SimpleTransform(config.count_feat_transform)),\n" % feat_name)
	        else:
	            f.write("('%s', SimpleTransform()),\n" % feat_name)