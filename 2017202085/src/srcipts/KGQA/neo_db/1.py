from datetime import datetime

class qmks:
    def __init__(self,ks_dict):
        self.ks={}
        for (k,v) in ks_dict.items():
            self.ks[k]=v


    def xzks(self,km,sj):
        self.ks[km] = sj
        print(len(self.ks))

    def ckks(self):
        for (k,v) in self.ks.items():
            date1 = datetime.strptime(v,'%Y/%m/%d %H:%M')
            date2 = datetime.strptime('2020/10/19 14:00','%Y/%m/%d %H:%M')
            if (date2- date1).days<0:
                print('%s的考试时间是%s,距离现在还剩%s天%s小时%s分'%(k,v,(date1-date2).days,(date1-date2).seconds/3600,(date1-date2).seconds/60))
            if (date2 - date1).days>=0:
                print('%s的考试时间是%s,距离现在已过%s天%s小时%s分' % (k, v, (date2 - date1).days, int((date2 - date1).seconds/3600) ,(date2 - date1).seconds%3600/60 ))

student1 = qmks(ks_dict={"sysjfx": "2020/10/19 8:30","sjwj": "2020/12/30 14:00"})
student2 = qmks(ks_dict={"sysjfx": "2020/10/19 8:30"})
student1.xzks('tjx','2020/12/12 14:00')
student1.ckks()

