

class exFeatures:


    def __init__(self,IdTweet,HasURLContent,StringCertin,fingGfwers,actions,URLOfLoc,StringCertainLoc,Retweet):
        self.__IdTweet = IdTweet
        self.__HasURLContent = HasURLContent
        self.__StringCertin = StringCertin
        self.__fingGfwers = fingGfwers
        self.__actions = actions
        self.__URLOfLoc = URLOfLoc
        self.__StringCertainLoc = StringCertainLoc
        self.__Retweet = Retweet

    # getters


    def getIdTweet(self):
        return self.__IdTweet
    def getHasURLContent(self):
        return self.__HasURLContent
    def getStringCertin(self):
        return self.__StringCertin
    def getfingGfwers(self):
        return self.__fingGfwers
    def getactions(self):
        return self.__actions
    def getURLOfLoc(self):
        return self.__URLOfLoc
    def getStringCertainLoc(self):
        return self.__StringCertainLoc
    def getRetweet(self):
        return self.__Retweet



    #setters

    def setIdTweet(self, IdTweet):
        self.__IdTweet = IdTweet

    def setHasURLContent(self, HasURLContent):
        self.__HasURLContent = HasURLContent

    def setStringCertin(self, StringCertin):
        self.__StringCertin = StringCertin

    def setfingGfwers(self, fingGfwers):
        self.__fingGfwers = fingGfwers

    def setactions(self, actions):
        self.__actions = actions

    def setURLOfLoc(self, URLOfLoc):
        self.__URLOfLoc = URLOfLoc

    def setStringCertainLoc(self, StringCertainLoc):
        self.__StringCertainLoc = StringCertainLoc

    def setRetweet(self, Retweet):
        self.__Retweet = Retweet


