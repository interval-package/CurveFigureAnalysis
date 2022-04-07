

class OutputErrorOfBlank(Exception):
    def __init__(self, msg):
        super(OutputErrorOfBlank, self).__init__("invalid of blank,  " +msg)


class OutputErrorOfSpecificPos(OutputErrorOfBlank):
    def __init__(self, msg):
        super(OutputErrorOfSpecificPos, self).__init__("point not found,  " +msg)




