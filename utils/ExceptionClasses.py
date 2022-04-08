

class OutputErrorOfBadQuality(Exception):
    def __init__(self, msg=''):
        super(OutputErrorOfBadQuality, self).__init__(msg)


class OutputErrorOfBlank(OutputErrorOfBadQuality):
    def __init__(self, msg):
        super(OutputErrorOfBlank, self).__init__("invalid of blank,  " +msg)


class OutputErrorOfSpecificPos(OutputErrorOfBlank):
    def __init__(self, msg):
        super(OutputErrorOfSpecificPos, self).__init__("point not found,  " +msg)




