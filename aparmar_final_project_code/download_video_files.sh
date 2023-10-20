mkdir -p input_files
cd input_files

curl -LOJ https://web.archive.org/web/20190901190223/http://www.nada.kth.se/cvap/actions/walking.zip
curl -LOJ https://web.archive.org/web/20190901190223/http://www.nada.kth.se/cvap/actions/jogging.zip
curl -LOJ https://web.archive.org/web/20190901190223/http://www.nada.kth.se/cvap/actions/running.zip
curl -LOJ https://web.archive.org/web/20190901190223/http://www.nada.kth.se/cvap/actions/boxing.zip
curl -LOJ https://web.archive.org/web/20190901190223/http://www.nada.kth.se/cvap/actions/handwaving.zip
curl -LOJ https://web.archive.org/web/20190901190223/http://www.nada.kth.se/cvap/actions/handclapping.zip

unzip *.zip
cd ../