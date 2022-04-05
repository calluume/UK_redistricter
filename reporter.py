from plotter import *

import cv2, time, os.path, csv
from os import remove as delete_frame
from datetime import datetime, timedelta

party_colours = {'conservatives': ['#0087DC', 'Blues'],
                 'labour': ['#DC241f', 'Reds'],
                 'liberal_democrats': ['#FDBB30', 'YlOrBr'],
                 'ukip': ['#33114B', 'Purples'],
                 'green': ['#6AB123', 'YlGn'],
                 'plaid_cymru': ['#338736', 'YlGn'],
                 'snp': ['#FFFF00', 'YlOrBr'],
                 'dup': ['#D4694B', 'OrRd'],
                 'sinn_fein': ['#00635E', 'Greens'],
                 'sdlp': ['#4DA165', 'YlGn'],
                 'uup': ['#48A6EE', 'Blues'],
                 'alliance': ['#CCAE2C', 'YlOrBr'],
                 'other_parties': ['#909090', 'Greys']          
                }

class Reporter:
    def __init__(self, plotter=None, create_log=True, progress_bar=False):
        self.no_func_evals = 0

        self.maps = []
        self.frame_text = []
        self.start_time = None
        self.plotter = plotter
        self.generate_progress_bar = progress_bar
        self.k = 0
        self.kmax = 0

        self.logs = []
        if create_log:
            now = datetime.now()
            self.log_filename = now.strftime("data/logs/log_%Y.%d.%m_%H.%M.%S.csv")
            self.log_file = open(self.log_filename, 'w')
            self.log_writer = csv.writer(self.log_file)
            header = ['k', 'func_eval', 'fitness', 'fairness', 'compactness', 'no_changes']
            for party in party_colours.keys():
                header.append(party+'_sv_diff')
            self.log_writer.writerow(header)
        else:
            self.log_writer = None

    def record_video_frame(self, wards, constituencies, k, text=None, images_file="images/video_frames", random_colours=None, frame_repeats=1):
        now = datetime.now()
        image_filename = "{0}/{1}_itr{2}.png".format(images_file, now.strftime("%Y.%d.%m_%H.%M.%S"), k+1)
        self.maps += [image_filename] * frame_repeats
        self.frame_text += [text] * frame_repeats
        self.plotter.plot_ward_party_support_map(wards, constituencies, metric='winner', value_type='constituency', image_savefile=image_filename, random_colours=random_colours)

    def generate_image_video(self, video_filename, length=None, delete_frames=True):

        video_filename = 'videos/' + video_filename
        if '.' in video_filename:
            if not video_filename.endswith('.mp4'):
                print('Class REPORTER ERROR: Invalid video filename: {0}\n  ↳ Filename must use \'.mp4\' format.'.format(video_filename))
                exit()
        else:
            video_filename += '.mp4'

        img_array = []

        for i in range(len(self.maps)):
            filename = self.maps[i]
            if not os.path.exists(filename):
                print('Class REPORTER ERROR: Video frame cannot be found: {0}\n  ↳ Image may have been deleted or had its name changed.'.format(video_filename))
                exit()
            elif not filename.endswith('.png'):
                print('Class REPORTER ERROR: Invalid filename found: {0}\n  ↳ Image frames must use \'.png\' format.'.format(video_filename))
                exit()

            img = cv2.imread(filename)
            height, width, _ = img.shape
            size = (width,height)

            text = self.frame_text[i]
            if text != None:
                cv2.putText(img, text, (int(height * 0.05), height - int(height * 0.05)), cv2.FONT_HERSHEY_DUPLEX, 5, (0, 0, 255), 6)
            
            img_array.append(img)

        if length != None: fps = len(img_array) / length
        else: fps = 1.5
        
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(get_unique_video_filename(video_filename), fourcc, fps, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

        if delete_frames:
            for frame in self.maps:
                delete_frame(frame)

    def update_stats(self, fitness_metrics, vote_shares=None, no_changes=None, k=None, stage_text=""):
        if k != None:
            self.k = k
        if self.log_writer != None:
            
            curr_log = [self.k, self.no_func_evals] + fitness_metrics

            if no_changes != None: curr_log.append(no_changes)
            else: curr_log.append('NULL')

            for party in party_colours.keys():
                curr_log.append((vote_shares[party][1] - vote_shares[party][3])/vote_shares[party][3])

            self.log_writer.writerow(curr_log)
            self.logs.append(curr_log)

        if self.generate_progress_bar:
            generate_progress_bar(self.k, self.kmax, 'k: {0}/{1} ({2}), fitness: {3} {4}'.format(self.k, self.kmax, str(timedelta(seconds=int(time.time() - self.start_time))), round(fitness_metrics[0], 6), stage_text), prefix='Progress: ')
        if self.k == self.kmax: print()

    def close(self, show_plot=True, plot_title=None, save_plot=None, verbose=True):
        """
        Once generation is over, stats are processed
        """

        if self.log_writer != None: self.log_file.close()
        end_time = time.time()
        if verbose:
            initial = self.logs[0]
            results = self.logs[-1]
            print("\nFinal result:\n  Fitness: {0} ({1:+})\n  Fairness: {2} ({3:+})\n  Compactness: {4} ({5:+})".format(round(results[2], 6), round(results[2] - initial[2], 3), round(results[3], 6), round(results[3] - initial[3], 3),round(results[4], 6), round(results[4] - initial[2], 3)))
            print("\n  Time: {0}\n  Iterations: {1} ({2} /itr)".format(str(timedelta(seconds=round(end_time - self.start_time))),
                                                                       self.k,
                                                                       str(timedelta(seconds=round((end_time - self.start_time)/(self.k))))))

        plot_performance(self.logs, show_plot=show_plot, title=plot_title, save_plot=save_plot)

        return
        
def get_unique_video_filename(video_filename):

    if not os.path.exists(video_filename):
        return video_filename
    else:
        filename_arr = video_filename.split('.')
        video_no = 0
        while os.path.exists(video_filename):
            video_no += 1
            video_filename = "{0} ({1}).{2}".format(filename_arr[0], video_no, filename_arr[1])
        
        return video_filename