AdaRank_CSharp.exe help
AdaRank_CSharp.exe train -data train.txt -model AdaRank.model -measure MAP -round 1000
AdaRank_CSharp.exe test -data train.txt -model AdaRank.model -output train.score -round 800
AdaRank_CSharp.exe eval -data train.txt -score train.score -output train.eval.400

