import cv2, os
from datetime import datetime as dt
import numpy as np

def createDir(path, isFolder):
    if isFolder and path[-1] != '/':
        path += '/'
    slashes = [p for p, c in enumerate(path) if c == '/']
    for slash in slashes:
        if not os.path.exists(path[0:slash+1]):
            os.mkdir(path[0:slash+1])
    return path

def run( videoIn, videoOut = '', frameOutDir = '', frameSamplingFreq = 0.):
    src = cv2.VideoCapture(videoIn)
    out = None
    tot = int(src.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(src.get(cv2.CAP_PROP_FPS))

    print("Processing %d frames in %s"%(tot, videoIn))
    if videoOut != '':
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(createDir(videoOut, False),fourcc, 60.02, (1920,1080))

    if frameOutDir != '' and frameSamplingFreq > 0:
        createDir(frameOutDir, True)
        frameSamplingFreq = min(frameSamplingFreq, fps)
        if videoIn is 0:
            framePrefix = 'camera'
        else:
            _, framePrefix = os.path.split(videoIn)
    else:
        frameSamplingFreq = 0

    ret, frame = src.read()
    if ret:
        frame = frame.astype("float")
        fore = frame.copy()
        back = frame.copy()
        frameCnt = 0
        showLine = 0

        while True:
            ret, frame = src.read()
            if not ret:
                break
            frame = frame.astype("float")
            frameCnt += 1
            
            cv2.accumulateWeighted(frame, fore, 0.75)
            cv2.accumulateWeighted(frame, back, 0.01)

            showDelta = cv2.absdiff(cv2.convertScaleAbs(fore),cv2.convertScaleAbs(back))
            if out:
                out.write(showDelta)
            if frameSamplingFreq and frameCnt % int(fps / frameSamplingFreq) is 0:
                cv2.imwrite(os.path.join(frameOutDir, "%s_%d.png"%(framePrefix,
                    frameCnt)), showDelta)
            
            if frameCnt > showLine * tot:
                showLine += 0.1
                print(dt.now(), "...finished %0.1f"%(frameCnt / tot))
        
    src.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    for root, dirs, files in os.walk('/home/raymondkwok/data/cloudchamber/'):
        for file in files:
            if file[-4:] == '.mp4':
                run( os.path.join(root, file), 
                     videoOut = os.path.join(root, file + '_deBkg.avi'), 
                     frameOutDir = os.path.join(root, file + '_frames'), 
                     frameSamplingFreq = 2)
