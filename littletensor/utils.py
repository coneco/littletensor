RESOURCES_KEY = "__resources"

def isinstancelist(obj, assert_type):
    if not isinstance(obj, list):
        return False
    for item in obj:
        if not isinstance(item, assert_type):
            return False
    return True
