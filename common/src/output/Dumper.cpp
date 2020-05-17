#include "common/include/output/Dumper.hpp"
#include "common/include/output/Logger.hpp"

#include <boost/filesystem.hpp>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace std;

Dumper::Dumper(const string& path, const string& dirname) {
    ostringstream outputFileStream;

    time_t t = time(NULL);
    tm* now = localtime(&t);

    outputFileStream << path << "/";
    outputFileStream << dirname << "_";

    outputFileStream << now->tm_mon + 1 << "_" << now->tm_mday << "_" << now->tm_hour << now->tm_min;

    outputFileStream << "/";

    outputDirectoryPath = outputFileStream.str();
}

void Dumper::createDir() const {
    if (!boost::filesystem::create_directory(outputDirectoryPath)) {
        throw invalid_argument("Directory couldn't be created");
    }
}

void Dumper::dumpAlgorithm(ComputeAlgorithm* algorithm) const {
    ofstream algorithmOutputFile;
    algorithmOutputFile.open(outputDirectoryPath + "algorithm.txt");

    if (algorithmOutputFile.is_open()) {

        algorithmOutputFile << "==== Compute Algorithm Summary ====" << endl;
        algorithmOutputFile << algorithm->toString() << endl;

        algorithmOutputFile.close();
    }
}

void Dumper::dumpGatherParameters(const string& file) const {
    Gather* gather = Gather::getInstance();

    ofstream gatherOutputFile;
    gatherOutputFile.open(outputDirectoryPath + "gather.txt");

    if (gatherOutputFile.is_open()) {

        gatherOutputFile << "==== Gather Summary ====" << endl;
        gatherOutputFile << "Input file = " << file << endl;
        gatherOutputFile << gather->toString() << endl;

        gatherOutputFile.close();
    }
}

void Dumper::dumpResult(const string& resultName, const MidpointResult& result) const {
    Gather* gather = Gather::getInstance();

    ofstream resultOutputFile;
    resultOutputFile.open(outputDirectoryPath + resultName + ".su");

    if (resultOutputFile.is_open()) {

        for (auto it = gather->getCdps().begin(); it != gather->getCdps().end(); it++) {

            float m0 = it->first;
            const Cdp& cdp = it->second;

            const vector<float>& samples = result.get(m0);

            trace_info_t cdpInfo = cdp.getCdpInfo();

            resultOutputFile.write(
                reinterpret_cast<const char*>(&cdpInfo),
                sizeof(trace_info_t)
            );

            resultOutputFile.write(
                reinterpret_cast<const char*>(samples.data()),
                samples.size() * sizeof(float)
            );
        }

        resultOutputFile.close();
    }
}

void Dumper::dumpStatisticalResult(const string& statResultName, const StatisticalMidpointResult& statResult) const {
    Gather* gather = Gather::getInstance();

    ofstream resultOutputFile;
    resultOutputFile.open(outputDirectoryPath + statResultName + ".csv");

    if (resultOutputFile.is_open()) {

        for (auto it = gather->getCdps().begin(); it != gather->getCdps().end(); it++) {

            float m0 = it->first;

            resultOutputFile << m0 << "," << statResult.getStatisticalResultForMidpoint(m0) << endl;
        }

        resultOutputFile.close();
    }
}

void Dumper::dumptraveltime(Traveltime* model) const {
    ofstream modelOutputFile;
    modelOutputFile.open(outputDirectoryPath + "traveltime.txt");

    if (modelOutputFile.is_open()) {

        modelOutputFile << "==== Model Summary ====" << endl;
        modelOutputFile << model->toString() << endl;

        modelOutputFile.close();
    }
}


// void Dumper::dumpExArguments(double totalElapsedTime) const {

//     ofstream argumentsOutputFile;
//     argumentsOutputFile.open(outputDirectoryPath + "/parameters.txt");

//     Arguments& const arguments = Arguments::getInstance();

//     if (argumentsOutputFile.is_open()) {

//         argumentsOutputFile << "** " << MethodTypeStrings[INDEX_OF(computeTechnology)];
//         argumentsOutputFile << " (" << MethodStrategyDescStrings[INDEX_OF(computeStrategy)] << ") **";
//         argumentsOutputFile << endl << endl;

//         argumentsOutputFile << "** Method " << MethodStrings[INDEX_OF(arguments.chosen_method)] << " **" << endl;
//         argumentsOutputFile << "** Common Parameters **" << endl;
//         argumentsOutputFile << "input data = " << arguments.input_file << endl;
//         argumentsOutputFile << "aph = " << arguments.aph << endl;
//         argumentsOutputFile << "tau = " << arguments.tau << endl;
//         argumentsOutputFile << "azimuth = " <<  arguments.azimuthInRadians << " rad" << endl;

//         if (computeStrategy != MethodStrategy::STACK) {

//             switch(arguments.chosen_method) {
//                 case Method::CMP:
//                     argumentsOutputFile << "vmin = " <<  arguments.vmin << endl;
//                     argumentsOutputFile << "vmax = " <<  arguments.vmax << endl;

//                     if (computeStrategy == MethodStrategy::EXHAUSTIVE ||
//                         computeStrategy == MethodStrategy::EXHAUSTIVE_GREEDY) {
//                         argumentsOutputFile << "nv = " <<  arguments.nv << endl;
//                     }
//                     break;

//                 case Method::CRS:
//                     argumentsOutputFile << endl << "** CRS parameters **" << endl << endl;
//                     argumentsOutputFile << "apm = " << arguments.apm << endl;
//                     argumentsOutputFile << "vmin = " << arguments.vmin << endl;
//                     argumentsOutputFile << "vmax = " << arguments.vmax << endl;
//                     argumentsOutputFile << "v0 = " << arguments.v0 << endl;
//                     argumentsOutputFile << "ain = " << arguments.ainInRadians << " rad" << endl;
//                     argumentsOutputFile << "bpctg = " << arguments.bpctg << endl;

//                     if (computeStrategy == MethodStrategy::EXHAUSTIVE ||
//                         computeStrategy == MethodStrategy::EXHAUSTIVE_GREEDY) {
//                         argumentsOutputFile << "nv = " << arguments.nv << endl;
//                         argumentsOutputFile << "na = " << arguments.na << endl;
//                         argumentsOutputFile << "nb = " << arguments.nb << endl;
//                     }
//                     break;

//                 case Method::CRP:
//                     argumentsOutputFile << endl << "** CRP parameters **" << endl << endl;
//                     argumentsOutputFile << "apm = " << arguments.apm << endl;
//                     argumentsOutputFile << "aph0 = " << arguments.aph0 << endl;
//                     argumentsOutputFile << "vmin = " << arguments.vmin << endl;
//                     argumentsOutputFile << "vmax = " << arguments.vmax << endl;
//                     argumentsOutputFile << "amin = " << arguments.amin << endl;
//                     argumentsOutputFile << "amax = " << arguments.amax << endl;
//                     argumentsOutputFile << "h0 = " << arguments.h0 << endl;

//                     if (computeStrategy == MethodStrategy::EXHAUSTIVE ||
//                         computeStrategy == MethodStrategy::EXHAUSTIVE_GREEDY) {
//                         argumentsOutputFile << "nv = " << arguments.nv << endl;
//                         argumentsOutputFile << "na = " << arguments.na << endl;
//                     }
//                     break;
//             }
//         }
//         else {
//             argumentsOutputFile << "nn = " << arguments.nn << endl;
//             argumentsOutputFile << "v_array = " << arguments.v_array << endl;

//             switch(arguments.chosen_method) {
//                 case Method::CRS:
//                     argumentsOutputFile << "b_array = " << arguments.b_array << endl;

//                 case Method::CRP:
//                     argumentsOutputFile << "a_array = " << arguments.a_array << endl;
//                     argumentsOutputFile << "apm = " << arguments.apm << endl;
//                     break;
//             }
//         }

//         if (arguments.cdp_min > 0) {
//             argumentsOutputFile << "cdpmin = " << arguments.cdp_min << endl;
//         }

//         if (arguments.cdp_max > 0) {
//             argumentsOutputFile << "cdpmax = " << arguments.cdp_max << endl;
//         }

//         if (computeStrategy == MethodStrategy::GA) {
//             argumentsOutputFile << endl << "** GA parameters **" << endl << endl;
//             argumentsOutputFile << "Generations = " << arguments.gen << endl;
//             argumentsOutputFile << "Individuals per population = " << arguments.pop_size << endl;
//         }

//         argumentsOutputFile << "duration = " << totalElapsedTime << " seconds" << endl;

//         argumentsOutputFile.close();
//     }
// }
