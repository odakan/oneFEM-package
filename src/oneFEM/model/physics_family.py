##-----------------------------------------------------------------------##
#                                                                         #
#        #--oneFEM--#: One FEM software in a galaxy far far away          #
#                                                                         #
#                   Computational Mechanics 2022                          #
#                       University of Pavia                               #
#               Written by: Onur Deniz AKAN, IUSS Pavia                   #
#                         15 January 2022                                 #
#                                                                         #
##-----------------------------------------------------------------------##
#
# Author: Onur Deniz Akan
# Date: 09/03/2026
# Version: 0.1
#
# PhysicsFamily enum — compatibility key between element, kinematics, and material.

from enum import Enum


class PhysicsFamily(Enum):
    CONTINUUM_CAUCHY       = "continuum/cauchy"
    CONTINUUM_COSSERAT     = "continuum/cosserat"
    CONTINUUM_MICROMORPHIC = "continuum/micromorphic"
    CONTINUUM_BIOT_CAUCHY  = "continuum/biot_cauchy"
    CONTINUUM_BIOT_COSSERAT = "continuum/biot_cosserat"
    CONTINUUM_BIOT_MICRO   = "continuum/biot_micromorphic"
    SHELL                  = "shell"
    BEAM                   = "beam"
    ZL                     = "zl"
    CONTACT                = "contact"
