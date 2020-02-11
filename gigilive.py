# -*- coding: UTF-8 -*-
# gigilive.py: main file for GiGiLive
###############################################################################
#
# This file is part of GiGiRedFilter.
#
#    Copyright (C) 2019 Andrea Console  <andreaconsole@gmail.com>
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#    If you find any part of this code useful and suitable for any software
#    developed by you, I would appreciate if you mentioned me in your code.
#
################################################################################
import logging
import sys
from astropy.io import fits
import numpy as np
import PyIndi
import png
import subprocess
from flask import Flask, render_template, request
import signal
import threading
import time

# local imports
import processing


# Inherting the base class 'Thread' to save files asynchronously
class FitsAsyncWrite(threading.Thread):
    def __init__(self, data, folderpath, suffix):
        # calling superclass init
        threading.Thread.__init__(self)
        self.data = data
        self.folderpath = folderpath
        self.suffix = suffix

    def run(self):
        filename = self.folderpath+"frame_"+time.strftime("%Y%m%d_%H%M%S", time.localtime())+self.suffix+".fit"
        # open a file and save buffer to disk
        with open(filename, "wb") as f:
            f.write(self.data)


def signal_handler(signal, frame):
    print ('You pressed Ctrl+C - or killed me with -2')
    # .... Put your logic here .....
    indiclient.disconnectServer()
    sys.exit(0)


class IndiClient(PyIndi.BaseClient):
    device = None

    def __init__(self):
        super(IndiClient, self).__init__()
        self.logger = logging.getLogger('PyQtIndi.IndiClient')
        self.logger.info('creating an instance of PyQtIndi.IndiClient')
        self.averagedframe = None
        self.calframe = None
        self.mode = "searching"

    def newDevice(self, d):
        self.logger.info("new device " + d.getDeviceName())
        if d.getDeviceName() == "CCD Simulator":
            self.logger.info("Set new device CCD Simulator!")
            # save reference to the device in member variable
            self.device = d

    def newProperty(self, p):
        self.logger.info("new property " + p.getName() + " for device " + p.getDeviceName())
        if self.device is not None and p.getName() == "CONNECTION" and p.getDeviceName() == self.device.getDeviceName():
            self.logger.info("Got property CONNECTION for CCD Simulator!")
            # connect to device
            self.connectDevice(self.device.getDeviceName())
            # set BLOB mode to BLOB_ALSO
            self.setBLOBMode(1, self.device.getDeviceName(), None)
        if p.getName() == "CCD_EXPOSURE":
            # take first exposure
            self.takeExposure()

    def removeProperty(self, p):
        self.logger.info("remove property " + p.getName() + " for device " + p.getDeviceName())

    def newBLOB(self, bp):
        self.logger.info("new BLOB " + bp.name)
        # get image data
        img = bp.getblobdata()
        # write image data to BytesIO buffer
        import io
        blobfile = io.BytesIO(img)
        # convert fit to numpy array
        hdulist = fits.open(blobfile)
        hdu = hdulist[0]
        image_array = hdu.data.astype(float)
        # extract luminance and color (from sensor RGB to linear Lab) TODO
        l_fit, a_fit, b_fit = processing.sensor2Lab(image_array)
        # the software has three operating modes:
        # searching: no sum nor dark substraction is performed. Fast mode for finding objects
        # calibrate: creates the dark frame for the live stacking. The mount must be turned off because it needs
        #            a moving sky in front of the camera to remove stars from the dark/flat calibration frame
        # livestacking: exponential moving average, dark/flat substraction
        if self.mode == "livestacking":
            #save fit to disk
            if indiclient.savefits:
                background = FitsAsyncWrite(blobfile.getvalue(), self.output_file_path_fit, 'L')
                background.start()
            if self.calframe is None:
                self.calframe = np.zeros(np.shape(l_fit))
            l_fit = l_fit - self.calframe + np.amax(self.calframe)
            # align and stack
            if self.averagedframe is None:
                self.averagedframe = l_fit
                self.logger.info("first image of the stack")
            shiftvalues = processing.register(self.averagedframe, l_fit, 100, 512)
            self.logger.info(">>>>> image shifted by: " + str(shiftvalues))
            # if the mount moved too much, reset self.averagedframe
            if max(shiftvalues[2], shiftvalues[3]) > self.maxshift:
                self.averagedframe = l_fit
                self.logger.info("frame not aligned - error bigger than limit")
            else:
                self.averagedframe = processing.imtrans(self.averagedframe, -shiftvalues[2], -shiftvalues[3])
            self.averagedframe = processing.expaverage(self.averagedframe, l_fit, self.avrgrate)
            l_fit = self.averagedframe
        elif self.mode == "calibrate":
            self.averagedframe = None
            if self.calframe is None:
                self.calframe = l_fit
            else:
                self.calframe = processing.calibrationaverage(self.calframe, l_fit)
            l_fit = self.calframe
        elif self.mode == "searching":
            self.averagedframe = None

        # stretch and convert averagedframe to 8 bit
        l_fit = processing.imstretch8(l_fit)
        # colorize the output image according to the fit color data TODO
        image_array_3D = processing.colorize(l_fit, a_fit, b_fit)
        # save image
        filepath = self.output_file_path_png+"frame.png"
        png.from_array(image_array_3D, 'L').save(filepath) # TODO remove L
        # start new exposure
        self.takeExposure()

    def newSwitch(self, svp):
        self.logger.info("new Switch " + svp.name + " for device " + svp.device)

    def newNumber(self, nvp):
        self.logger.info("new Number " + nvp.name + " for device " + nvp.device)

    def newText(self, tvp):
        self.logger.info("new Text " + tvp.name + " for device " + tvp.device)

    def newLight(self, lvp):
        self.logger.info("new Light " + lvp.name + " for device " + lvp.device)

    def newMessage(self, d, m):
        # self.logger.info("new Message "+ d.messageQueue(m))
        pass

    def serverConnected(self):
        print("Server connected (" + self.getHost() + ":" + str(self.getPort()) + ")")

    def serverDisconnected(self, code):
        self.logger.info("Server disconnected (exit code = " + str(code) + "," + str(self.getHost()) + ":" + str(
            self.getPort()) + ")")

    def takeExposure(self):
        self.logger.info(">>>>>>>>")
        # get current exposure time
        exp = self.device.getNumber("CCD_EXPOSURE")
        # set exposure time to 5 seconds
        exp[0].value = self.exptime
        self.logger.info("exposure time = {0} second/s".format(self.exptime))
        # send new exposure time to server/device
        self.sendNewNumber(exp)


app = Flask(__name__)

# manages the controls sent from the owner's web page (change of mode and exp. length)
@app.route('/update_values')
def add_numbers():
    mode = request.args.get('a', 0, type=int)
    if mode == 1:
        indiclient.mode = "searching"
    elif mode == 2:
        indiclient.mode = "calibrate"
        indiclient.calframe = None
    elif mode == 3:
        indiclient.mode = "livestacking"
    else:
        indiclient.mode = "searching"
    indiclient.exptime = request.args.get('b', 0, type=int)
    return 'received'


# web page for the owner
@app.route('/adminsecretpage')  # you can rename it to keep it "secret"
def master():
    return render_template('admin.html', staticfilepath = indiclient.output_file_path_png,
                           exptime = indiclient.exptime)


# web page for the audience
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler) #bind to program exit to stop the server
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    # instantiate the client
    indiclient = IndiClient()
    indiclient.output_file_path_png = "./static/images/"
    #
    #
    # ---------------------------- user input below
    indiclient.exptime = 4  # exposure time
    indiclient.maxshift = 50  # max number of pixel shift before the software assumes it is a new target
    indiclient.avrgrate = 0.2  # moving average rate (0<rate<1; if rate=1 you will see only the last captured frame)
    indiclient.savefits = True # do you want to save every fit while in livestacking mode?
    indiclient.output_file_path_fit = "./fits/" # where to save fit files
    indiserver_drivers = ["indi_simulator_telescope", "indi_simulator_ccd"] # server to use
    # ---------------------------- user input above
    #
    #
    # set indi server localhost and port 7624
    indiclient.setServer("localhost", 7624)
    # connect to indi server
    print("Connecting to indiserver")
    indiservercommand = ["indiserver"] + indiserver_drivers
    proc = subprocess.Popen(indiservercommand)
    time.sleep(5)
    if not (indiclient.connectServer()):
        print("No indiserver running on " + indiclient.getHost() + ":" + str(indiclient.getPort()) + " - Try to run")
        print("  indiserver indi_simulator_telescope indi_simulator_ccd")
        sys.exit(1)



    # start the web server
    app.run(host='0.0.0.0')
