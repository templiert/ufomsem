import os, sys, time

import smtplib
import base64
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np

def get_metrics_graph_paths(working_folder):
    metrics_folder = os.path.normpath(working_folder)
    metrics_folder = os.path.join(
        working_folder,
        'section_metrics')

    round_metrics_folders = sorted([
        f.path for f in os.scandir(metrics_folder)
        if f.is_dir()])

    plots_folder = os.path.join(working_folder, 'metrics_plots')
    if not os.path.isdir(plots_folder):
        os.mkdir(plots_folder)

    n_rounds = len(round_metrics_folders)
    n_sections = len(os.listdir(round_metrics_folders[0]))

    n_rows = int(n_sections**0.5)
    n_cols = n_sections//n_rows
    if n_rows*n_cols < n_sections:
        n_rows+=1

    focus_scores = []
    intensity_means = []
    intensity_stdevs= []

    for id_section in range(n_sections):
        focus_scores_section = []
        intensity_means_section = []
        intensity_stdevs_section = []

        for id_round in range(n_rounds):
            metrics_path = os.path.join(
                metrics_folder,
                'metrics_round_' + str(id_round).zfill(3),
                'section_metrics_' + str(id_round).zfill(3) + '_' + str(id_section).zfill(3) + '.txt')

            with open(metrics_path, 'r') as f:
                focus, intensity_mean, intensity_stdev = list(map(float, f.readline().split('\t')))
                if focus == -999:
                    focus = 0
                if intensity_mean == -999:
                    intensity_mean = 0
                if intensity_stdev == -999:
                    intensity_stdev = 0

                focus_scores_section.append(focus)
                intensity_means_section.append(intensity_mean)
                intensity_stdevs_section.append(intensity_stdev)

        focus_scores.append(focus_scores_section)
        intensity_means.append(intensity_means_section)
        intensity_stdevs.append(intensity_stdevs_section)

    plot_paths = []

    msem_folder = os.path.join(
        os.path.dirname(working_folder),
        'msem')

    round_folders = sorted([
        f.path for f in os.scandir(msem_folder)
        if f.is_dir() and ('Logs' not in os.path.basename(f))])

    section_names = [
        x[4:]
        for x in sorted(os.listdir(round_folders[0]))
        if os.path.isdir(
            os.path.join(
                round_folders[0],
                x))]

    for metric, val in zip(
        ['Focus score', 'Intensity mean', 'Intensity stdev'],
        [focus_scores, intensity_means, intensity_stdevs]):

        fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
        ax = fig.add_subplot(111, frameon=False)
        ax.set_xlabel('Imaging rounds', fontsize = 10)
        ax.set_ylabel(metric, fontsize = 10)
        ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        fig.suptitle(
            metric + ' (plot y-axes) of the '
            + str(n_sections) + ' sections (plot grid) '
            + '\nacross ' + str(n_rounds) + ' imaging rounds (plot x-axes)',
            fontsize=10)

        for id_section in range(n_sections):

            try:
                fig.axes[id_section].scatter(
                    np.arange(n_rounds),
                    np.array(val[( id_section+5) % n_sections ]),
                    marker='o',
                    s=1)

                fig.axes[id_section].set_title(
                    section_names[( id_section+5) % n_sections], size = 8,
                    x=0.35 ,y=0.9)
                    # x = 0.15, y = -0.07)

            except Exception as e:
                print('exception ', e)

        plot_path = os.path.join(
            plots_folder,
            metric.replace(' ', '_') + '.jpg')

        plot_paths.append(plot_path)
        fig.savefig(
            plot_path,
            bbox_inches='tight',
            dpi = 100)

    return plot_paths

def encode(t):
    return base64.b64encode(t).encode('utf-8')
def decode(t):
    return base64.b64decode(t).decode('utf-8')

def send_email(
    sender,
    recipients,
    subject,
    message,
    im_paths,
    working_folder):

    # for x in [sender, recipients, subject, message, im_paths, working_folder]:
        # x = decode(x)

    sender, recipients, subject, message, im_paths, working_folder = list(map(decode, [sender, recipients, subject, message, im_paths, working_folder]))

    # sender =
    # recipients = decode(recipients)
    # subject = decode(subject)
    # message = decode(message)
    # im_paths = decode(im_paths)
    # working_folder = decode(working_folder)

    with open(os.path.join(working_folder, 'test.txt'), 'w') as f:
        f.write(im_paths)
        f.write(working_folder)

    if ',' in im_paths:
        im_paths = im_paths.split(',')
    else:
        im_paths = [im_paths]

    im_paths = im_paths + get_metrics_graph_paths(working_folder)

    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = subject
    msgRoot['From'] = sender
    msgRoot['To'] = recipients
    msgRoot.preamble = 'This is a multi-part message in MIME format.'

    text_insert_images = ''
    for id, im_path in enumerate(im_paths):
        if os.path.isfile(im_path):
            text_insert_images += '<img src="cid:image' + str(id) + '"><br><br>'

    msgText = MIMEText(
            text_insert_images
            + '<p style="font-family: monospace;font-size:12px;">'
            + message
            + '</p>',
            'html')
    msgRoot.attach(msgText)

    for id, im_path in enumerate(im_paths):
        if os.path.isfile(im_path):
            with open(im_path, 'rb') as fp:
                msgImage = MIMEImage(fp.read())
            msgImage.add_header('Content-ID', '<image' + str(id) + '>')
            msgRoot.attach(msgImage)

    s = smtplib.SMTP(
        'smtp.office365.com',
        587)
    s.starttls()
    s.login(
        sender,
        (base64.b64decode('xxx')
            .decode('utf-8')))
    s.sendmail(
        sender,
        recipients.split(','),
        msgRoot.as_string())
    s.quit()

if __name__ == '__main__':
    send_email(*sys.argv[1:])