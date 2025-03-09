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
# Date: 01/03/2025
# Version: 0.1
#
#MATERIAL main object definition
#   definition of a FE material object and functions

class Material(object):
    def __init__(self, mat_id=-1):
        self.__ID = mat_id

    def _getClassType(self):
        return "Material"

    def _setTrialStrain(self, strain):
        self.__eps = strain
        self.__f = self.__C * self.__eps

    # get state
    def getStrain(self):
        return self.__eps

    def getStress(self):
        return self.__f

    def getTangent(self):
        return self.__C

    def getInitialTangent(self):
        return self.__C0

    # handle state
    def commitState(self):
        self.__f_commit = self.__f
        self.__eps_commit = self.__eps
        return 0

    def revertToLastCommit(self):
        self.__f = self.__f_commit
        self.__eps=self.__eps_commit
        return 0

    def revertToStart(self):
        self.__C = self.__C0
        self.__f = 0.0
        self.__f_commit = 0.0
        self.__eps = 0.0
        self.__eps_commit = 0.0
        return 0

    ## copy and others...
    #UniaxialMaterial* getCopy(void);
    #void Print(OPS_Stream& s, int flag = 0);
#
    ## send/recv self
    #int sendSelf(int commitTag, Channel& theChannel);
    #int recvSelf(int commitTag, Channel& theChannel, FEM_ObjectBroker& theBroker);
#
    ## parameters and responses
    #int setParameter(const char** argv, int argc, Parameter& param);
    #int updateParameter(int parameterID, Information& info);
    #Response* setResponse(const char** argv, int argc, OPS_Stream& output);
    #int getResponse(int responseID, Information& matInformation);
    #double getEnergy(void);