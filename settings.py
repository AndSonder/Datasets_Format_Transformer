# coding=utf8

# this dirt is the id of classes, you can change this to make transformer produce different result
class2id = {
    'person': 0
}

# the smallest id of classes in the dirt class2id
START_BOUNDING_BOX_ID = 0

id2class = dict(zip(class2id.values(), class2id.keys()))
