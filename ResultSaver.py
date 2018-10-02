import datetime


class ResultSaver(object):
    """class that saves results of OMR recognizion"""

    def __init__(self, file_name):
       self.file_name = file_name
       f = open(self.file_name, "w")
       now = datetime.datetime.now()
       f.write((str(now))[:(str(now)).rfind('.')])
       f.write('\n-----------------------\n')
       f.close()
       

    def write(self, result, img_name):
      f = open(self.file_name, "a")
      f.write(str(img_name))
      f.write('\n-----------------------\n')
      for row in result:
          f.write(str(row) + '\n')
      f.write('-----------------------\n')
      f.close()

