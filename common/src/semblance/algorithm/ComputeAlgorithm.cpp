#include "common/include/semblance/algorithm/ComputeAlgorithm.hpp"

#include <numeric>

using namespace std;

ComputeAlgorithm::ComputeAlgorithm(
    const string& name,
    shared_ptr<Traveltime> model,
    shared_ptr<DeviceContext> context,
    DataContainerBuilder* dataBuilder
) : algorithmName(name),
    deviceContext(context),
    traveltime(model),
    dataFactory(dataBuilder),
    threadCount(1024),
    threadCountToRestore(threadCount),
    computedStatisticalResults(static_cast<unsigned int>(StatisticResult::CNT)) {
}

ComputeAlgorithm::~ComputeAlgorithm() {
}

void ComputeAlgorithm::copyGatherDataToDevice() {
    Gather* gather = Gather::getInstance();

    unsigned int traceCount = gather->getTotalTracesCount();

    deviceFilteredTracesDataMap[GatherData::MDPNT].reset(dataFactory->build(traceCount, deviceContext));

    deviceFilteredTracesDataMap[GatherData::HLFOFFST].reset(dataFactory->build(traceCount, deviceContext));

    vector<float> tempMidpointArray(traceCount), tempHalfoffsetArray(traceCount);

    for (unsigned int t = 0; t < traceCount; t++) {
        tempMidpointArray[t] = gather->getMidpointOfTrace(t);
        tempHalfoffsetArray[t] = gather->getHalfoffsetOfTrace(t);
    }

    deviceFilteredTracesDataMap[GatherData::MDPNT]->copyFrom(tempMidpointArray);
    deviceFilteredTracesDataMap[GatherData::HLFOFFST]->copyFrom(tempHalfoffsetArray);

    deviceFilteredTracesDataMap[GatherData::FILT_SAMPL].reset(dataFactory->build(deviceContext));

    deviceFilteredTracesDataMap[GatherData::FILT_MDPNT].reset(dataFactory->build(deviceContext));

    deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST].reset(dataFactory->build(deviceContext));
}

float ComputeAlgorithm::getStatisticalResult(StatisticResult statResult) const {
    if (computedStatisticalResults.find(statResult) == computedStatisticalResults.end()) {
        throw invalid_argument("Couldn't find value for statistical result.");
    }

    return computedStatisticalResults.at(statResult);
}


void ComputeAlgorithm::saveStatisticalResults(
    unsigned long totalUsedTracesCount,
    chrono::duration<double> totalExecutionTime
) {
    Gather* gather = Gather::getInstance();

    vector<float> tempNotUsedCount(deviceNotUsedCountArray->getElementCount());

    deviceNotUsedCountArray->pasteTo(tempNotUsedCount);

    unsigned long notUsedTracesCount, interpolationsPerformed;

    notUsedTracesCount = static_cast<unsigned long>(
        accumulate(tempNotUsedCount.begin(), tempNotUsedCount.end(), 0.0f)
    );

    interpolationsPerformed = static_cast<unsigned long>(gather->getWindowSize()) *
        (totalUsedTracesCount - notUsedTracesCount);

    computedStatisticalResults[StatisticResult::INTR_PER_SEC] =
        static_cast<float>(interpolationsPerformed) / static_cast<float>(totalExecutionTime.count());

    computedStatisticalResults[StatisticResult::EFFICIENCY] =
        1.0f - static_cast<float>(notUsedTracesCount) / static_cast<float>(totalUsedTracesCount);
}

void ComputeAlgorithm::copyOnlySelectedTracesToDevice(
    const vector<unsigned char>& usedTraceMask
) {
    Gather* gather = Gather::getInstance();

    unsigned int traceCount = gather->getTotalTracesCount();

    filteredTracesCount = accumulate(usedTraceMask.begin(), usedTraceMask.end(), 0);

    /* Reallocate filtered sample array */
    deviceFilteredTracesDataMap[GatherData::FILT_SAMPL]->reallocate(filteredTracesCount * gather->getSamplesPerTrace());
    deviceFilteredTracesDataMap[GatherData::FILT_MDPNT]->reallocate(filteredTracesCount);
    deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST]->reallocate(filteredTracesCount);

    vector<float> tempMidpoint(filteredTracesCount);
    vector<float> tempHalfoffset(filteredTracesCount);

    unsigned int cudaArrayOffset = 0, idx = 0;
    for (unsigned int i = 0; i < traceCount; i++) {
        if (usedTraceMask[i]) {

            const Trace& trace = gather->getTraceAtIndex(i);

            deviceFilteredTracesDataMap[GatherData::FILT_SAMPL]->copyFromWithOffset(trace.getSamples(), cudaArrayOffset);

            tempMidpoint[idx] = trace.getMidpoint();
            tempHalfoffset[idx] = trace.getHalfoffset();

            cudaArrayOffset += gather->getSamplesPerTrace();
            idx++;
        }
    }

    deviceFilteredTracesDataMap[GatherData::FILT_MDPNT]->copyFrom(tempMidpoint);
    deviceFilteredTracesDataMap[GatherData::FILT_HLFOFFST]->copyFrom(tempHalfoffset);
}

void ComputeAlgorithm::changeThreadCountTemporarilyTo(unsigned int t) {
    threadCount = t;
}

void ComputeAlgorithm::restoreThreadCount() {
    threadCount = threadCountToRestore;
}
