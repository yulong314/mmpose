from dis import dis
import math
import copy
import os.path

import numpy as np
import time
import datetime
import cv2
import json
import shutil

from collections import defaultdict
from pycocotools.cocoeval import COCOeval
# from detectron2.data import DatasetCatalog, MetadataCatalog
from mmpose.utils.logger import get_root_logger
import logging
from functools import partial
# import matplotlib.pyplot as plt
class PixelDisEval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='keypoints', dataset_info = None, img_prefix = None, anndir = None,isdraw_gt_dt = True):
        super().__init__(cocoGt, cocoDt, 'keypoints')
        self.logger = get_root_logger()
        self.logger.propagate = False
        self.dataset_info = dataset_info
        self.setKeypointInfo(dataset_info)
        # self.keypointNames = [value['name'] for key, value in dataset_info.keypoint_info.items()]
        # self.pose_kpt_color = [value['color'] for key, value in dataset_info.keypoint_info.items()]
        self.img_prefix = img_prefix
        subfolder = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.folder = os.path.join(anndir, "evalImages/", subfolder)
        self.folder = self.folder.replace("../", "")
        self.anndir = anndir
        self.gtdrawCnt = 0
        self.isdraw_gt_dt = isdraw_gt_dt
        self.evalImgsByCam = defaultdict(list)
        self.camCnt = 14
        for i in range(self.camCnt):
            self.evalImgsByCam[i] = []
        self.meanByCam = np.zeros(self.camCnt)
        self.mean95ByCam = np.zeros(self.camCnt)
        self.stdByCam = np.zeros(self.camCnt)
        self.std95ByCam = np.zeros(self.camCnt)
    def setKeypointInfo(self, datasetInfo):
        # from mmpose.datasets import DatasetInfo
        # self.dataset_info = DatasetInfo(datasetInfo)
        self.dataset_info = datasetInfo
        # keypointsInfo = self.datasetInfo['keypoints']
        # self.pose_kpt_color = [value['color'] for key, value in datasetInfo.items()]
        # self.keypoint_name = [value['name'] for key, value in datasetInfo.items()]
        self.skeleton = self.dataset_info.skeleton
        pose_link_color = self.dataset_info.pose_link_color
        pose_kpt_color= self.dataset_info.pose_kpt_color
        self.pose_kpt_color = []
        self.pose_link_color = []
        for i in range(len(pose_kpt_color)):
            color = tuple(int(c) for c in pose_kpt_color[i])
            self.pose_kpt_color.append(color) 
        for i in range(len(pose_link_color)):
            color = tuple(int(c) for c in pose_link_color[i])
            self.pose_link_color.append(color) 
        self.keypointNames = self.dataset_info.keypoint_name

    def shiftTextPosition(self, position, imgshape):
        if position[0] > .8 * imgshape[1]:
            position[0] = 0
        if position[1] < .1 * imgshape[0]:
            position[1] = int(.2 * imgshape[0]) 
    def checkTxtPosition(self,ps,p1)            :
        result = p1
        for p in ps:
            if abs(result - p ) < 20:
                result += 30
                # if result > 350:
                #     result = 30
        return result



    # def _getSavePath(self,annid, filename, saveFolder):
    #     filename1 = filename.replace("../", "")
    #     filename1 = filename1.replace("/", "_")
    #     # savepath = os.path.join(saveFolder, filename1)
    #     savepath = os.path.join(saveFolder,f'{annid:03d}_{filename1}')
    #     return savepath
    def getImgname(self, imgId):
        imgname = self.cocoDt.imgs[imgId]['file_name']
        w = self.cocoDt.imgs[imgId]['width']
        h = self.cocoDt.imgs[imgId]['height']
        return imgname,w,h

    def getSavename(self, imgname, imgId, gtid, subdir = ''):
        split = os.path.basename(imgname).split('.')
        stem = split[0]
        ext = split[1]
        # stem.replace('/', '_')
        saveName1 = f'{stem}_{str(imgId)}_{str(gtid)}.{ext}'
        # if  self.imgid_largeErrors and imgId in self.imgid_largeErrors:
        #     saveName = os.path.join(os.path.dirname(imgname), 'largeErrors', saveName1)
        # else:
        saveName = os.path.join(os.path.dirname(imgname), subdir,saveName1)

        saveName=saveName.replace('../', '')
        saveName=saveName.replace('/', '_')
        return saveName

    def computeOks1(self, imgId, catId):
        p = self.params
        imgname, width,height = self.getImgname(imgId)
        saveName = self.getSavename(imgname, imgId, 0)
        # dimention here should be Nxm
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
        dts = [dts[i] for i in inds]
        if len(dts) > p.maxDets[-1]:
            dts = dts[0:p.maxDets[-1]]
        # if len(gts) == 0 and len(dts) == 0:

        # ious = np.zeros((len(dts), len(gts)))
        ious = [[np.array([-10000,-10000]) for x in range(len(gts))] for y in range(len(dts))]
        if len(gts) == 0 or len(dts) == 0:
            return ious
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2)**2
        k = len(sigmas)
        # assert len(gts) == len(dts)
        # compute oks between each detection and ground truth object
        confThreshold = 0.3
        camIndex = int(imgname.split('/')[-2]) - 1
        for j, gt in enumerate(gts):
            if max(gt['keypoints']) == 0:
                continue
            if 'num_keypoints' in gt and gt['num_keypoints'] == 0:
                continue
            g = np.array(gt['keypoints']).reshape(-1,3)

            for i, dt in enumerate(dts):
                d = np.array(dt['keypoints']).reshape(-1,3)
                conf = (g[:,2] > 0) * (d[:,2] > 0)
                distances = np.linalg.norm(g[:, :2] - d[:, :2], axis=1) * conf / width * 1920
                score = np.sum(distances)
                falsPositive = (g[:,2] == 0) * (d[:,2] >= confThreshold)
                trueNegative = (g[:,2] == 0) * (d[:,2] < confThreshold)
                falseNegative = (g[:,2] > 0) * (d[:,2] < confThreshold)
                truePositive = (g[:,2] > 0) * (d[:,2] >= confThreshold)
                confidence = d[:, 2]
                distMax = np.max(distances)
                maxIndex = np.argmax(distances)
                ious[i][j] = (score, distances, trueNegative, falsPositive, falseNegative, truePositive, gt['bbox'],saveName,confidence,distMax,maxIndex,camIndex)
        return ious

    def evaluateImg(self, imgId, catId, maxDet):
        '''
        perform evaluation for single category and image
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId,catId]
            dt = self._dts[imgId,catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
        if len(gt) == 0 and len(dt) ==0:
            return None
        imgname,w,h = self.getImgname(imgId)
        for g in gt:
            # if g['ignore'] or (g['area']<aRng[0] or g['area']>aRng[1]):
            #     g['_ignore'] = 1
            # else:
                g['_ignore'] = 0

        # sort dt highest score first, sort gt ignore last
        # gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        # gt = [gt[i] for i in gtind]
        # assert len(gt) == gtind.shape[0]
        # dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        # assert dtind.shape[0] == len(dt)
        # dt = [dt[i] for i in dtind[0:maxDet]]
        iscrowd = [int(o['iscrowd']) for o in gt]
        # load computed ious
        # ious = [self.ious[imgId, catId][i] for i in gtind] if len(self.ious[imgId, catId]) > 0 else self.ious[imgId, catId]
        ious = self.ious[imgId, catId]
        T = len(p.iouThrs)
        T = 1
        G = len(gt)
        D = len(dt)
        gtm  = np.zeros((T,G),dtype = np.int32)
        dtm  = np.zeros((T,D),dtype = np.int32)
        gtIg = np.array([g['_ignore'] for g in gt])
        dtIg = np.zeros((T,D))
        iou = []
        scores = []
        filenames = []
        if not len(ious)==0:
            # for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    min_dist = 1e10
                    m   = -1
                    for gind, g in enumerate(gt):
                        # if this gt already matched, and not a crowd, continue
                        if gtm[0,gind]>0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        # if m>-1 and gtIg[m]==0 and gtIg[gind]==1:
                        #     break
                        # continue to next gt unless better match made
                        score = ious[dind][gind][0]
                        if score >= min_dist:
                            continue
                        # if match successful and best so far, store appropriately
                        min_dist=score
                        m=gind
                    # if match made store id of match for both dt and gt
                    if m ==-1:
                        continue
                    dtIg[0,dind] = gtIg[m]
                    dtm[0,dind]  = m
                    gtm[0,m]     = dind
                    # if dind != m:
                    #     print("dind != m", dind, m)
                        # assert False
                    iou.append(ious[dind][m])
                    saveName = self.getSavename(imgname, imgId, m)  
                    filenames.append(saveName)
                    # iou.append(ious[dind][dind])
                    # scores.append(min_dist)
        # store results for given image and category
        
              
        return {
                'image_id':     imgId,
                'category_id':  catId,
                'dtIds':        [d['id'] for d in dt],
                'gtIds':        [g['id'] for g in gt],
                'dtMatches':    dtm,
                'gtMatches':    gtm,
                # 'dtScores':     scores,
                # 'gtIgnore':     gtIg,
                # 'dtIgnore':     dtIg,
                'iou':          iou,
                'filenames':    filenames,
                'imgname':      imgname,
                # 'cam':          int(imgname.split('/')[-2]) - 1,
            }                

    def __moveLargeErrorImgs(self, outpaths, kpIndexes_to_draw = None):
        # if kpIndexes_to_draw is None:
        if outpaths is None:
            return
        for outpath, imgId in outpaths:
            if imgId in self.imgid_largeErrors:
                order = self.imgid_largeErrors.index(imgId)
                dir = os.path.dirname(outpath)
                basename = os.path.basename(outpath)
                orderedName = f"{order:03d}_{basename}"
                largeerror_dir = os.path.join(dir, 'largeErrors')
                if not os.path.exists(largeerror_dir):
                    os.makedirs(largeerror_dir,exist_ok= True)
                largeErrorPath = os.path.join(largeerror_dir, orderedName)
                shutil.move(outpath, largeErrorPath)

    def _drawevalImage(self, evalInfo, kpIndexes_to_draw,largeErrorOnly):
        # def draw_gt_dt(self, imgId, catId ):
        if evalInfo is None:
            return

        outpaths = self.draw_gt_dt(evalInfo, kpIndexes_to_draw,largeErrorOnly)
        # self.__moveLargeErrorImgs(outpaths, kpIndexes_to_draw)

    def __draw_gt_dts(self, evalImgs, kpIndexes_to_draw = None, largeErrorOnly = False, parallel = False):
        # return
        # self.logger.info(f"draw_gt_dts {kpIndexes_to_draw} ...")
        partialFun = partial(self._drawevalImage, kpIndexes_to_draw = kpIndexes_to_draw, largeErrorOnly = largeErrorOnly)
        if largeErrorOnly:
            errors = self.largeErrors
        else:
            errors  = evalImgs
        if parallel:
            import multiprocessing as mp
            pool = mp.Pool(32)
            pool.map(partialFun, errors)
        else:
            for evalInfo in errors:
                partialFun(evalInfo)
        # self.logger.info(f"draw_gt_dts {kpIndexes_to_draw} done")

        pass

    def calcPresicionRecall(self,truePositives, falsePositives, falseNegatives):
        # distance, falsePositive, falseNegative, truePositive, trueNegative
        # truePositives = np.sum(npiou[:,:,3])
        # falseNegatives = np.sum(npiou[:,:,2])
        # falsePositives = np.sum(npiou[:,:,1])
        precision = truePositives/(truePositives + falsePositives)
        recall = truePositives/(truePositives + falseNegatives)
        return precision, recall
    def draw_gt_dt(self, evalInfo, kpIndexes_to_draw = None,largeErrorOnly = False ):
        outpaths = []
        if self.img_prefix is None:
            return outpaths
        imgId = evalInfo['image_id']
        if largeErrorOnly:
            if imgId not in self.imgid_largeErrors:
                return outpaths
        catId = evalInfo['category_id']            
        filenames = evalInfo['filenames']
        gts = self._gts[imgId, catId]
        dts = self._dts[imgId, catId]
        
        if len(dts) == 0:
            # print('warning: len(gt) == 0')
            return
        gtm = evalInfo['gtMatches']
        dtm = evalInfo['dtMatches']
        gtids = evalInfo['gtIds']
        dtids = evalInfo['dtIds']

        imgname,w,h = self.getImgname(imgId)
        # savePath= os.path.join(self.img_prefix, saveName)
        imgpath = os.path.join(self.img_prefix, imgname)
        img1 = cv2.imread(imgpath)
        assert img1 is not None

        confThres = 0.1
        
        for dtIndex in range(len(dts)):
            img = img1.copy()
            # gtId = gtids[i]
            gtId = dtm[0][dtIndex]

            # dtId = gtm[0][dtIndex]
            gt = gts[gtId]
            dt = dts[dtIndex]
            # assert gt['id'] == gtId
            # assert dt['id'] == dtId


            kpgts = np.array(gt['keypoints']).reshape(-1, 3)
            kpdts = np.array(dt['keypoints']).reshape(-1, 3)
            cnt = 0
            pose_kpt_color = self.pose_kpt_color#[(255,0, 255),(0,255,0),(255,255,0),(0,255,255),(255,0,0), (0, 0, 255), (255,255,255), (128,0, 128)]* 2
            radius = 3#int(img.shape[0] / 1520 * 10 + 1)  // 2 + 1
            txtScale = int(img.shape[0] / 1520 * 2) + 1
            thickness =  int(img.shape[0] / 1520 * 2) + 1    
            txtpositiony = []
            if kpIndexes_to_draw is None:
                kpIndexes_to_draw = range(0,kpgts.shape[0])
                subdir_kp = 'all'
            else:
                subdir_kp = self.keypointNames[kpIndexes_to_draw[0]]
            for i in kpIndexes_to_draw:
                g = kpgts[i]
                d = kpdts[i]
                # print(d[2],str(d[2]))
                
                if g[2] > 0:
                    cv2.circle(img, (int(g[0] + 0.5), int(g[1] + 0.5)), radius, pose_kpt_color[i])
                    distance = evalInfo['iou'][dtIndex][1][i]
                    distancesmall =  np.linalg.norm(g[:2] - d[:2])
                    confidenceStr = f"{d[2]:.2f}:{distance:.1f}:{self.keypointNames[i]}"
                else:
                    confidenceStr = f"{d[2]:.2f}:nan:{self.keypointNames[i]}"
                # if d[2] < confThres:
                #     continue            
                detectPosition = [int(d[0] + 0.5), int(d[1] + 0.5)]
                txtpositiony1 = int(d[1])
                txtpositiony1 = self.checkTxtPosition(txtpositiony,txtpositiony1)
                txtpositiony.append(txtpositiony1)
                textPosition = [int(d[0]), txtpositiony1]
                cv2.circle(img, detectPosition, 2*radius, pose_kpt_color[i])
                
                # self.shiftTextPosition(textPosition,img.shape)
                
                cv2.putText(img, confidenceStr, textPosition,1, 1, pose_kpt_color[i],1)
                cnt = 1-cnt
            preds = kpdts
            for sk_id, sk in enumerate(self.skeleton):
                color1 = self.pose_link_color[sk_id]
                if sk[0] in kpIndexes_to_draw and sk[1] in kpIndexes_to_draw:
                    pos1p = (int(preds[sk[0], 0]), int(preds[sk[0], 1]))
                    pos2p = (int(preds[sk[1], 0]), int(preds[sk[1], 1]))
                    conf1 = preds[sk[0], 2]
                    conf2 = preds[sk[1], 2]
                    if conf1 > confThres and conf2 > confThres:
                        cv2.line(img, pos1p, pos2p, color1, thickness=1)
            maxErrorKpIndex= evalInfo['iou'][0][10]
            maxErrorKpName= self.keypointNames[maxErrorKpIndex]
            maxDistance = evalInfo['iou'][0][9]
            camIndex = evalInfo['iou'][0][11]
            if self.cam:
                assert  camIndex == self.cam
            saveName = self.getSavename(imgname, imgId, gtId)
            subdir_cam = f"cam{camIndex + 1}" if  self.cam else "all"
            outpath= os.path.join(self.folder, subdir_cam, subdir_kp ,saveName)
            outpath = outpath.replace("../","")
            dir = os.path.dirname(outpath)
            if not os.path.exists(dir):
                os.makedirs(dir,exist_ok= True)
            # outpath = outpath.replace('png','jpg')
            box = gt['bbox']
            if box[0] < 0:
                box[0] = 0
            cv2.rectangle(img, (int(box[0]), int(box[1]) ), (int(box[0] + box[2]),int(box[1] +box[3])), (255,255,255))
            imgdraw = img[ int(box[1]) : int(box[1] +box[3]+100), int(box[0]):int(box[0] + box[2])+500]
            cv2.putText(imgdraw, f"{maxErrorKpName}:{maxDistance}", (10, 30),1, 1, pose_kpt_color[maxErrorKpIndex],1)
            assert imgdraw is not None and imgdraw.shape[0] > 0 and imgdraw.shape[1] > 0, f"{imgdraw.shape} {box}"
            cv2.imwrite(outpath, imgdraw)
            # cv2.imshow("img",imgdraw)
            # cv2.waitKey(0)
            # if self.gtdrawCnt == 0:
            #     print(f"{outpath} saved")
            outpaths.append((outpath, imgId))
            self.gtdrawCnt += 1            
        return outpaths
    def accumulate(self, p = None):
        # self.drawIousHist()
        
        
        '''
        Accumulate per image evaluation results and store the result in self.eval
        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')

        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]

        _pe = self._paramsEval
        setI = set(_pe.imgIds)


        i_list = [n for n, i in enumerate(p.imgIds)  if i in setI]
        
        # score, distances, trueNegative, falsPositive, falseNegative, truePositive = [], [], [], [], [], []
        itemCnt = len(self.evalImgs[0]['iou'][0])#11
        alist = [[],[],[],[],[],[],[],[],[],[],[],[]]
        assert itemCnt == len(alist)
        alistByCam=[]
        for i in range(self.camCnt):
            alistByCam.append([[],[],[],[],[],[],[],[],[],[],[],[]])
        for i in i_list:
            if self.evalImgs[i] is None:
                continue
            evalImg = self.evalImgs[i]['iou']
            imgpath = self.evalImgs[i]['imgname']
            split = imgpath.split('/')
            cam = split[-2]
            camInt = int(cam) - 1
            self.evalImgsByCam[camInt].append(self.evalImgs[i])
            for k in range(len(evalImg)):
                for j in range(itemCnt):
                    alist[j].append(evalImg[k][j])
                    alistByCam[camInt][j].append(evalImg[k][j])
                
        self.evalByCam = [{
            'distances': alistByCam[i][1],
            'trueNegative': alistByCam[i][2],
            'falsPositive': alistByCam[i][3],
            'falseNegative': alistByCam[i][4],
            'truePositive': alistByCam[i][5],
            'imgname': alistByCam[i][7],
            'confidence': alistByCam[i][8],
            'distMax': alistByCam[i][9],
            'maxIndex': alistByCam[i][10],            
        } for i in range(self.camCnt)]
        self.eval = {
            # 'params': p,
            # 'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            # 'score': alist[0],
            'distances': alist[1],
            'trueNegative': alist[2],
            'falsPositive': alist[3],
            'falseNegative': alist[4],
            'truePositive': alist[5],
            'imgname': alist[7],
            'confidence': alist[8],
            'distMax': alist[9],
            'maxIndex': alist[10],
            # 'all':alist
        }
        toc = time.time()
        print('accumulate DONE (t={:0.2f}s).'.format( toc-tic))    

    def summarize(self):
        
        # for  cam in range(self.camCnt):
        #     self.logger.info(f"************* cam:{cam + 1} **************************")
        #     self.summarize_evals(self.evalImgsByCam[cam], self.evalByCam[cam], cam)
        self.logger.info(f"************* cam:all**************************")            
        self.summarize_evals(self.evalImgs, self.eval)
        # evalsByCams = self.divideByCam(self.evalImgs)
        # for cam, evals in evalsByCams.items():
        #     self.logger.info(f"************* cam:{cam} **************************")
        #     self.summarize_evals(evals)

    def summarize_evals(self,evalImgs,evals,cam = None):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        self.cam = cam
        np.set_printoptions(linewidth=np.inf,precision=2)
        p = self.params
        iStr = ' {} of accupoint {} = {:0.3f}  std= {:0.3f}  mean95= {:0.3f} '
        kpIndexes_to_draw = None
        byCame = False
        self.__sortEvalImgs(evalImgs, kpIndexes_to_draw)
        self.__staticLargeErrors(kpIndexes_to_draw)

        distances = np.array(evals['distances'])
        trueNegative = np.array(evals['trueNegative'])
        falsPositive = np.array(evals['falsPositive'])
        falseNegative = np.array(evals['falseNegative'])
        truePositive = np.array(evals['truePositive'])
        confidence = np.array(evals['confidence'])

        distances[~truePositive] = 0

        mean_cnt = np.count_nonzero(distances)
        mean_all = np.mean(distances, where=distances>0)
        mean = np.mean(distances, where=distances>0,axis=0)
        max = np.max(distances, axis=0)
        maxIndexes = np.argmax(distances, axis=0)

        std = np.std(distances, where=distances>0,axis=0)
        std_all = np.std(distances, where=distances>0)
        distances_sort = np.sort(distances[distances>0])
        distances95 = distances_sort[0:int(distances_sort.shape[0]*0.95)]
        mean95_all = np.mean(distances95)
        std95_all = np.std(distances95)
        truePositive_joint = np.sum(truePositive, axis=0)
        falseNegative_joint = np.sum(falseNegative, axis=0)
        falsePositive_joint = np.sum(falsPositive, axis=0)
        trueNegative_joint = np.sum(trueNegative, axis=0)
        total_cnt = truePositive_joint + falseNegative_joint + falsePositive_joint + trueNegative_joint

        truePositive = np.sum(truePositive)
        falseNegative = np.sum(falseNegative)
        falsePositive = np.sum(falsPositive)
        if cam is not None:
            self.meanByCam[cam] = mean_all
            self.stdByCam[cam] = std_all
            self.mean95ByCam[cam] = mean95_all
            self.std95ByCam[cam] = std95_all
        if not evals:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType

        precision_joint, recall_joint = self.calcPresicionRecall(truePositive_joint, falsePositive_joint, falseNegative_joint)
        precision, recall = self.calcPresicionRecall(truePositive, falsePositive, falseNegative)
        self.stats = [1/mean95_all, 1/mean_all,precision, recall]
        # means = evals['meanIous']
        # mean = np.mean(means)
        # self.logger.info(f"keypointNames:{self.keypointNames}")
        
        self.logger.info(f"mean, mean95:{mean_all}, {mean95_all}")
        self.logger.info(f"precision,recall:{precision}, {recall}")
        self.logger.info(f" name, mean1, std1, 5%cnt, max1,  imgname, confidence")
        for i, (name, mean1, std1, percent5cnt,max1, maxIndex) in \
            enumerate(zip(self.keypointNames, mean, std, self.largeErrorCnt, max,maxIndexes)):
            self.logger.info(f"{name:20s}:{mean1:6.1f} {std1:6.1f} {percent5cnt} {max1:6.1f} : {evals['imgname'][maxIndex]:50s}   {confidence[maxIndex][i]:4.2f} ")
        if cam is None:
            self.logger.info(f"mean, std,  mean95, std95:{mean_all}, {std_all}, {mean95_all}, {std95_all}")
            for i in range(self.camCnt):
                self.logger.info(f"cam:{i + 1} :{self.meanByCam[i]:6.1f} {self.stdByCam[i]:6.1f} {self.mean95ByCam[i]:6.1f}, {self.std95ByCam[i]:6.1f} ")

        
        self.logger.info(f"truePositive:{truePositive_joint}")
        self.logger.info(f"falsePositive:{falsePositive_joint}")
        self.logger.info(f"falseNegative:{falseNegative_joint}")
        self.logger.info(f"precision_joint:{precision_joint}")
        self.logger.info(f"recall_joint:{recall_joint}")
        # self.logger.info(f"trueNegative:{trueNegative_joint}")
        # self.logger.info(f"total:{total_cnt}")
        self.logger.info(f"stats:{np.array(self.stats)}")
        
        if self.isdraw_gt_dt:
            self.__draw_gt_dts(evalImgs, kpIndexes_to_draw)

            kpIndexes_to_draw = range(len(self.keypointNames))
            # partialFun = partial(self.__draw_gt_dts_bigerrors)
            # import multiprocessing as mp
            # pool = mp.Pool(32)
            # pool.map(self._draw_gt_dts_bigerrors, kpIndexes_to_draw)            
            
            for i in  kpIndexes_to_draw:
                self._draw_gt_dts_bigerrors(evalImgs, i)
                # kpIndexes_to_draw = [i]
                # self.__sortEvalImgs(kpIndexes_to_draw)
                # self.__draw_gt_dts(kpIndexes_to_draw, largeErrorOnly = True)

    def _draw_gt_dts_bigerrors(self, evalImgs, kpIndex_to_draw):
        # return
        # self.logger.info(f"draw_gt_dts {kpIndex_to_draw} ...")
        self.__sortEvalImgs(evalImgs, [kpIndex_to_draw])
        self.__draw_gt_dts(evalImgs,kpIndexes_to_draw = [kpIndex_to_draw], largeErrorOnly = True, parallel = False)
        # self.logger.info("draw_gt_dts {kpIndexes_to_draw} done")

        pass
    def __compareEvalImgs(self, evalImgs1, kpIndexes_to_draw = None):
        x = evalImgs1['iou'][0]
        if kpIndexes_to_draw is None:
            maxerror = x[9]
            truepositive = x[5][x[10]]
        else:
            assert  len(kpIndexes_to_draw) == 1
            maxerror = x[1][kpIndexes_to_draw[0]]
            truepositive = x[5][kpIndexes_to_draw[0]]
        key = maxerror if truepositive else 0
        return key
    def __staticLargeErrors(self,kpIndexes_to_draw = None):
        #static number of large errors by kpindexes

        self.largeErrorCnt = np.zeros(len(self.keypointNames), dtype=np.int32)
        for lerror in self.largeErrors:
            if kpIndexes_to_draw is None:
                maxIndex = lerror['iou'][0][10]
            else:
                maxIndex = kpIndexes_to_draw[0]
            self.largeErrorCnt[maxIndex] += 1
        pass

    def __sortEvalImgs(self, evalImgs, kpIndexes_to_draw = None):
            # 'distances': alist[1],
            # 'trueNegative': alist[2],
            # 'falsPositive': alist[3],
            # 'falseNegative': alist[4],
            # 'truePositive': alist[5],
            # 'imgname': alist[7],
            # 'confidence': alist[8],        
        partialFun = partial(self.__compareEvalImgs, kpIndexes_to_draw = kpIndexes_to_draw)
        evalImgs.sort(key=partialFun, reverse=True)
        errorThreshold = 10.0
        cnt = len(evalImgs)
        for i in range(cnt):
            if partialFun(evalImgs[i]) < errorThreshold:
                break
        cnt05 = i
        self.largeErrors = evalImgs[0:cnt05]
        self.imgid_largeErrorsKPindexes = [{x['image_id']:[]} for x in self.largeErrors]
        self.imgid_largeErrors  = [x['image_id'] for x in self.largeErrors]
        pass

    def __str__(self):
        self.summarize()


    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif 'keypoints' in p.iouType:
            computeIoU = self.computeOks1
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, maxDet)
                 for catId in catIds
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('fuction evaluate DONE (t={:0.2f}s).'.format(toc-tic))

if __name__ == '__main__':
    rng = np.random.default_rng()
    mu, sigma = 0, 0.1
    ious = rng.normal(mu, sigma, 1000000)
    # ious = np.random.random(100) * 300
    percent = [50, 60, 80, 90, 95]
    
    percentile = np.percentile(ious, percent)
    for i in range(len(percent)):
        print(f"percentile {percent[i]}: {percentile[i]}" )

    amax = ious.max()
    amin = ious.min()
    numbins = int ((amax) / 2)
    hist, bin_edges = np.histogram(ious, bins= numbins)
    maxindex = np.argmax(hist)

    median = np.median(ious)
    print("median:",median)

    print("maxindex", maxindex)
    print("maxvalue", (bin_edges[maxindex] + bin_edges[maxindex + 1]) /2 )
    # plt.hist(ious, bins='auto')
    # plt.show()    
