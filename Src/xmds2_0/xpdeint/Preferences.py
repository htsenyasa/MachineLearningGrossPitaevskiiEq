#!/usr/bin/env python
# encoding: utf-8
import os

versionString = '2.2.3 "It came from the deep"'

if 'XMDS_USER_DATA' in os.environ:
    xpdeintUserDataPath = os.environ['XMDS_USER_DATA']
else:
    xpdeintUserDataPath = os.path.join(os.path.expanduser('~'), '.xmds')
