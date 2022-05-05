#!/usr/bin/env python
# coding: utf-8

# Python program to convert .mov files to .mp4 files, and remove original mov file
import os
    

def main():
    # assign directory
    directory = os.getcwd() #'target directory'

    # iterate over files in that directory
    for fdir, dirs, files in os.walk(directory):
        for fname_fext in files:
#            print(os.path.join(fdir, fname_fext))
            fpath = os.path.join(fdir, fname_fext)
            fdir_fname, fext = os.path.splitext(fpath)

            if fname_fext.endswith('.MOV') and not os.path.exists(os.path.join(fdir_fname+ '.MP4')):

                fname = fdir_fname.split('/')[-1]  # use for clear demo
                print(f">>> ffmpeg -i {fname}.MOV -vcodec h264 -acodec aac {fname}.MP4")
                os.system(f"ffmpeg -i '{fdir_fname}.MOV' -vcodec h264 -acodec aac '{fdir_fname}.MP4'")  
                print(f'>>> {fname}.mp4 converted')


    for fdir, dirs, files in os.walk(directory):
        for fname_fext in files:
            
            fpath = os.path.join(fdir, fname_fext)
            fdir_fname, fext = os.path.splitext(fpath)

            mov_path = os.path.join(fdir_fname+ '.MOV')
            mp4_path = os.path.join(fdir_fname+ '.MP4')
            if (os.path.exists(mov_path) and os.path.exists(mp4_path)): # 轉換成功後，兩者.mov 和.mp4 檔案都有，才把原本.mov 刪除
                os.remove(mov_path)     
                print(f'>>> {mov_path} has removed')


if __name__ == '__main__':
    main()
