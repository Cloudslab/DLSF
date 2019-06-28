package org.cloudbus.cloudsim.power;

import org.apache.commons.math3.util.MathUtils;
import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.CloudSimTags;
import org.cloudbus.cloudsim.core.SimEvent;
import org.cloudbus.cloudsim.core.predicates.PredicateType;
import org.cloudbus.cloudsim.lists.VmList;
import org.cloudbus.cloudsim.plus.util.CustomLog;
import org.cloudbus.cloudsim.util.MathUtil;
import org.python.core.PyObject;
import org.python.core.PyString;
import org.python.util.PythonInterpreter;

import java.util.*;
import java.util.logging.Level;

/**
 * The DRLDatacenter class implements functions for interaction
 * with the Deep Learning model and extends the PowerDatacenter class
 *
 * @author Shreshth Tuli
 * @since CloudSim Toolkit 5.0
 */

public class DRLDatacenter extends PowerDatacenter {

    public DRLDatacenterBroker broker;

    private double[] hostEnergy;

    private double savedCurrentTime;

    private double savedLastTime;

    private double savedTimeDiff;

    private double numVmsEnded;

    private double totalResponseTime;

    private double totalMigrationTime;

    private int InputLimit = 100;

    /**
     * Instantiates a new DRLDatacenter.
     *
     * @param name               the datacenter name
     * @param characteristics    the datacenter characteristics
     * @param schedulingInterval the scheduling interval
     * @param vmAllocationPolicy the vm provisioner
     * @param storageList        the storage list
     * @throws Exception the exception
     */
    public DRLDatacenter(
            String name,
            DatacenterCharacteristics characteristics,
            VmAllocationPolicy vmAllocationPolicy,
            List<Storage> storageList,
            double schedulingInterval,
            DRLDatacenterBroker broker) throws Exception {
        super(name, characteristics, vmAllocationPolicy, storageList, schedulingInterval);
        this.broker = broker;
        this.hostEnergy = new double[this.getHostList().size()];
    }

    protected void updateDLModel(){
        PythonInterpreter interpreter = ((DRLVmAllocationPolicy) getVmAllocationPolicy()).interpreter;
//        PyString loss = interpreter.eval("DeepRL().backprop(" + getLoss() + ")");
        PyString loss = (PyString)interpreter.eval("DeepRL().sendMap(" + getVmHostMap() + ")");
        System.out.println("DL Loss = " + ((PyObject) loss).toString());
        interpreter.eval("DeepRL().setInput(" + getInputMap() + ")");
    }

    public String getLoss(){
        String loss = "";
        double totalDataCenterEnergy = 0.0;
        double totalDataCenterCost = 0.0;
        for(DRLHost host : this.<DRLHost>getHostList()){
            totalDataCenterEnergy += this.hostEnergy[getHostList().indexOf(host)];
            totalDataCenterCost += (host.getCostModel().getCostPerCPUtime() * this.savedTimeDiff * host.getUtilizationOfCpu()) / (60 * 60);
        }
        loss = loss + "CurrentTime\t" + this.savedCurrentTime + "\n";
        loss = loss + "LastTime\t" + this.savedLastTime + "\n";
        loss = loss + "TimeDiff\t" + this.savedTimeDiff + "\n";
        loss = loss + "TotalEnergy\t" + totalDataCenterEnergy + "\n";
        loss = loss + "NumVsEnded\t" + this.numVmsEnded + "\n";
        loss = loss + "AverageResponseTime\t" + this.totalResponseTime/this.numVmsEnded + "\n";
        loss = loss + "AverageMigrationTime\t" + this.totalMigrationTime/this.numVmsEnded + "\n";
        this.numVmsEnded = 0; this.totalMigrationTime = 0; this.totalResponseTime = 0;
        loss = loss + "TotalCost\t" + totalDataCenterCost + "\n";
        loss = loss + "SLAOverall\t" + getSlaOverall(this.getVmList()) + "\n";
        if(getVmAllocationPolicy().getClass().getName().equals("DRLVmAllocationPolicy")){
            loss = loss + "HostPenalty\t" + ((DRLVmAllocationPolicy) getVmAllocationPolicy()).hostPenalty + "\n";
            loss = loss + "MigrationPenalty\t" + ((DRLVmSelectionPolicy)
                    ((DRLVmAllocationPolicy)
                            getVmAllocationPolicy()).getVmSelectionPolicy()).migrationPenalty + "\n";
        }
        return loss;
    }

    public String getInput(){
        String input = "";
        input = input + "CNN " + "number of VMs " + this.getVmList().size() + "\n";
        String temp;
        for(PowerVm vm : this.<PowerVm>getVmList()){
            temp = "";
            PowerHost host = (PowerHost)vm.getHost();
            temp = temp + ((host != null) ? (this.getHostList().indexOf(host)) : "NA")  + "\t";
            temp = temp + vm.getNumberOfPes() + "\t";
            temp = temp + MathUtil.sum(vm.getCurrentRequestedMips()) + "\t";
            temp = temp + vm.getCurrentRequestedMaxMips() + "\t";
            temp = temp + vm.getUtilizationMean() + "\t";
            temp = temp + vm.getUtilizationVariance() + "\t";
            temp = temp + vm.getSize() + "\t";
            temp = temp + vm.getCurrentAllocatedSize() + "\t";
            temp = temp + vm.getRam() + "\t";
            temp = temp + vm.getCurrentAllocatedRam() + "\t";
            temp = temp + vm.getCurrentRequestedRam() + "\t";
            temp = temp + vm.getBw() + "\t";
            temp = temp + vm.getCurrentAllocatedBw() + "\t";
            temp = temp + vm.getCurrentRequestedBw() + "\t";
            temp = temp + vm.getDiskBw() + "\t";
            temp = temp + vm.getCurrentAllocatedDiskBw() + "\t";
            temp = temp + vm.getCurrentRequestedDiskBw() + "\t";
            temp = temp + vm.isInMigration() + "\t";
            ArrayList<Vm> list = new ArrayList<Vm>(Arrays.asList(vm));
            temp = temp + getSlaOverall(list) + "\t";
            temp = temp + ((host != null) ? (host.getUtilizationOfCpuMips()) : "NA")  + "\t";
            temp = temp + ((host != null) ? (host.getAvailableMips()) : "NA")  + "\t";
            temp = temp + ((host != null) ? (host.getRamProvisioner().getAvailableRam()) : "NA")  + "\t";
            temp = temp + ((host != null) ? (host.getBwProvisioner().getAvailableBw()) : "NA")  + "\t";
            temp = temp + ((host != null) ? (host.getDiskBwProvisioner().getAvailableDiskBw()) : "NA")  + "\t";
            temp = temp + ((host != null) ? (host.getVmList().size()) : "0")  + "\t";
            temp = temp + ((host != null) ? (this.hostEnergy[getHostList().indexOf(vm.getHost())]) : "NA")  + "\t";
            input = input + temp + "\n";
        }
        input = input + "LSTM\n";
        for(DRLHost host : this.<DRLHost>getHostList()){
            temp = "";
            temp = temp + this.hostEnergy[getHostList().indexOf(host)] + "\t";
            temp = temp + host.getPower() + "\t";
            temp = temp + host.getMaxPower() + "\t";
            temp = temp + (host.getCostModel().getCostPerCPUtime() * this.savedTimeDiff / (60 * 60)) + "\t";
            temp = temp + getSlaOverall(host.getVmList()) + "\t";
            temp = temp + host.getUtilizationOfCpu() + "\t";
            temp = temp + host.getMaxUtilization() + "\t";
            temp = temp + host.getNumberOfPes() + "\t";
            temp = temp + host.getRamProvisioner().getAvailableRam() + "\t";
            temp = temp + host.getRamProvisioner().getRam() + "\t";
            temp = temp + host.getBwProvisioner().getAvailableBw() + "\t";
            temp = temp + host.getBwProvisioner().getBw() + "\t";
            temp = temp + host.getDiskBwProvisioner().getAvailableDiskBw() + "\t";
            temp = temp + host.getDiskBwProvisioner().getDiskBw() + "\t";
            temp = temp + host.getVmsMigratingIn().size() + "\t";
            input = input + temp + "\n";
        }
        return input;
    }

    @Override
    protected void updateCloudletProcessing() {
        if (getCloudletSubmitted() == -1 || getCloudletSubmitted() == CloudSim.clock()) {
            CloudSim.cancelAll(getId(), new PredicateType(CloudSimTags.VM_DATACENTER_EVENT));
            schedule(getId(), getSchedulingInterval(), CloudSimTags.VM_DATACENTER_EVENT);
            return;
        }
        double currentTime = CloudSim.clock();

        // if some time passed since last processing
        if (currentTime > getLastProcessTime()) {
            double minTime = this.updateCloudetProcessingWithoutSchedulingFutureEventsForce();
            System.out.println((int)currentTime/3600 + " hr " + ((int)(currentTime/60)-(60*((int)currentTime/3600))) + " min - " + getVmList().size());

            if (!isDisableMigrations()) {
                List<Map<String, Object>> migrationMap = getVmAllocationPolicy().optimizeAllocation(
                        getVmList());

                if (migrationMap != null) {
                    for (Map<String, Object> migrate : migrationMap) {
                        DRLVm vm = (DRLVm) migrate.get("vm");
                        PowerHost targetHost = (PowerHost) migrate.get("host");
                        PowerHost oldHost = (PowerHost) vm.getHost();

//                        if (oldHost == null) {
//                            Log.formatLine(
//                                    "%.2f: Migration of VM #%d to Host #%d is started",
//                                    currentTime,
//                                    vm.getId(),
//                                    targetHost.getId());
//                        } else {
//                            Log.formatLine(
//                                    "%.2f: Migration of VM #%d from Host #%d to Host #%d is started",
//                                    currentTime,
//                                    vm.getId(),
//                                    oldHost.getId(),
//                                    targetHost.getId());
//                        }

                        targetHost.addMigratingInVm(vm);
                        incrementMigrationCount();

                        /** VM migration delay = RAM / bandwidth **/
                        // we use BW / 2 to model BW available for migration purposes, the other
                        // half of BW is for VM communication
                        // around 16 seconds for 1024 MB using 1 Gbit/s network
                        send(
                                getId(),
                                vm.getRam() / ((double) targetHost.getBw() / (2 * 8000)),
                                CloudSimTags.VM_MIGRATE,
                                migrate);
                        vm.totalMigrationTime += (vm.getRam() / ((double) targetHost.getBw() / (2 * 8000)));
                    }
                }
            }

            // schedules an event to the next time
            if (minTime != Double.MAX_VALUE) {
                CloudSim.cancelAll(getId(), new PredicateType(CloudSimTags.VM_DATACENTER_EVENT));
                send(getId(), getSchedulingInterval(), CloudSimTags.VM_DATACENTER_EVENT);
            }

            setLastProcessTime(currentTime);
        }
    }


    /**
     * Update cloudet processing without scheduling future events.
     *
     * @return expected time of completion of the next cloudlet in all VMs of all hosts or
     *         {@link Double#MAX_VALUE} if there is no future events expected in this host
     */
    @Override
    protected double updateCloudetProcessingWithoutSchedulingFutureEventsForce() {
        double currentTime = CloudSim.clock();
        double minTime = Double.MAX_VALUE;
        double timeDiff = currentTime - getLastProcessTime();
        double timeFrameDatacenterEnergy = 0.0;

        this.savedCurrentTime = currentTime;
        this.savedLastTime = getLastProcessTime();
        this.savedTimeDiff = timeDiff;

//        Log.printLine("\n\n--------------------------------------------------------------\n\n");
//        Log.formatLine("New resource usage for the time frame starting at %.2f:", currentTime);

        for (DRLHost host : this.<DRLHost> getHostList()) {
//            Log.printLine();

            double time = host.updateVmsProcessing(currentTime); // inform VMs to update processing
            if (time < minTime) {
                minTime = time;
            }

//            Log.formatLine(
//                    "%.2f: [Host #%d] utilization is %.2f%%",
//                    currentTime,
//                    host.getId(),
//                    host.getUtilizationOfCpu() * 100);
        }

        if (timeDiff > 0) {
//            Log.formatLine(
//                    "\nEnergy consumption for the last time frame from %.2f to %.2f:",
//                    getLastProcessTime(),
//                    currentTime);

            for (PowerHost host : this.<PowerHost> getHostList()) {
                double previousUtilizationOfCpu = host.getPreviousUtilizationOfCpu();
                double utilizationOfCpu = host.getUtilizationOfCpu();
                double timeFrameHostEnergy = host.getEnergyLinearInterpolation(
                        previousUtilizationOfCpu,
                        utilizationOfCpu,
                        timeDiff);
                timeFrameDatacenterEnergy += timeFrameHostEnergy;

                this.hostEnergy[this.getHostList().indexOf(host)] = timeFrameHostEnergy;

//                Log.printLine();
//                Log.formatLine(
//                        "%.2f: [Host #%d] utilization at %.2f was %.2f%%, now is %.2f%%",
//                        currentTime,
//                        host.getId(),
//                        getLastProcessTime(),
//                        previousUtilizationOfCpu * 100,
//                        utilizationOfCpu * 100);
//                Log.formatLine(
//                        "%.2f: [Host #%d] energy is %.2f W*sec",
//                        currentTime,
//                        host.getId(),
//                        timeFrameHostEnergy);
            }

//            Log.formatLine(
//                    "\n%.2f: Data center's energy is %.2f W*sec\n",
//                    currentTime,
//                    timeFrameDatacenterEnergy);
        }

        setPower(getPower() + timeFrameDatacenterEnergy);

        checkCloudletCompletion();

        /** Remove completed VMs **/
        for (DRLHost host : this.<DRLHost> getHostList()) {
            for (Vm vm : host.getCompletedVms()) {
                if(currentTime - ((DRLVm)vm).startTime < 2 || vm.getCloudletScheduler().getCloudletExecList().size() < 1){
                    // Skip VMs just added
                    continue;
                }
                processVMDestroy(vm);
            }
        }

        /** Remove unallocated VMs **/
        for(Vm vm : getVmList()){
            if(vm.getHost() == null){
                getVmAllocationPolicy().deallocateHostForVm(vm);
                getVmList().remove(vm);
//                this.broker.getVmList().remove(vm);
                Log.printLine("VM destroyed = VM #" + vm.getId());
                Log.printLine("VMs left = " + getVmList().size());
            }
        }

        /** If VMs > input limit remove more **/
        int numRemove = getVmList().size()-this.InputLimit;
        for(int i = 0; i < Math.max(0, numRemove); i++){
            Vm vm = getVmList().get(i);
            getVmAllocationPolicy().deallocateHostForVm(vm);
            getVmList().remove(vm);
//          this.broker.getVmList().remove(vm);
            Log.printLine("VM destroyed = VM #" + vm.getId());
            Log.printLine("VMs left = " + getVmList().size());
        }


        Log.printLine();

        // Send reward and next input to DL Model
        if(getVmAllocationPolicy().getClass().getName().equals("DRLVmAllocationPolicy") && this.savedTimeDiff > 200){
            updateDLModel();
        }

//        Log.setDisabled(false);
        if(this.savedTimeDiff > 200){
            Log.printLine2("LOSS : \n" + getLoss());
            Log.printLine2(getVmHostMap());
            Log.printLine2("INPUT : \n" + getInput());
        }
//        Log.setDisabled(true);

        setLastProcessTime(currentTime);
        return minTime;
    }

    public String getVmHostMap(){
        String map = "";
        for(Vm vm : this.getVmList()){
            map = map + ("VM #" + vm.getId() + " index " + this.getVmList().indexOf(vm) + " <-> Host #" + vm.getHost().getId()) + "\n";
        }
        return map;
    }

    public String getInputMap(){
        String map = "";
        for(Vm vm : this.getVmList()){
            map = map + (this.getVmList().indexOf(vm) + " " + vm.getHost().getId()) + "\n";
        }
        return map;
    }

    public void processVMDestroy(Vm vm) {
        int vmId = vm.getId();

        // Remove the vm from the created list
        broker.getVmsCreatedList().remove(vm);
        broker.finilizeVM(vm);

        // Kill all cloudlets associated with this VM
        for (Cloudlet cloudlet : broker.getCloudletSubmittedList()) {
            if (!cloudlet.isFinished() && vmId == cloudlet.getVmId()) {
                try {
                    vm.getCloudletScheduler().cloudletCancel(cloudlet.getCloudletId());
                    cloudlet.setCloudletStatus(Cloudlet.FAILED_RESOURCE_UNAVAILABLE);
                } catch (Exception e) {
                    CustomLog.logError(Level.SEVERE, e.getMessage(), e);
                }

                sendNow(cloudlet.getUserId(), CloudSimTags.CLOUDLET_RETURN, cloudlet);
            }
        }

        // Use the standard log for consistency ....
        Log.printConcatLine(CloudSim.clock(), ": ", getName(), ": VM #", vmId,
                " has been destroyed in Datacenter #", this.getId());
        Log.printLine("VM #" + vm.getId() + " has been deallocated from host #" + vm.getHost().getId() + "with total reponse time " + ((DRLVm)vm).totalResponseTime + " and migration time " + ((DRLVm)vm).totalMigrationTime);
        this.totalResponseTime = ((DRLVm)vm).totalResponseTime;
        this.totalMigrationTime = ((DRLVm)vm).totalMigrationTime;
        this.numVmsEnded += 1;
        getVmAllocationPolicy().deallocateHostForVm(vm);
        getVmList().remove(vm);
        this.broker.getVmList().remove(vm);
        Log.printLine("VMs left = " + getVmList().size());
    }

    /**
     * Gets the sla metrics.
     *
     * @param vms the vms
     * @return the sla metrics
     */
    protected static double getSlaOverall(List<Vm> vms) {
        Map<String, Double> metrics = new HashMap<String, Double>();
        List<Double> slaViolation = new LinkedList<Double>();
        double totalAllocated = 0;
        double totalRequested = 0;

        for (Vm vm : vms) {
            double vmTotalAllocated = 0;
            double vmTotalRequested = 0;
            double vmUnderAllocatedDueToMigration = 0;
            double previousTime = -1;
            double previousAllocated = 0;
            double previousRequested = 0;
            boolean previousIsInMigration = false;

            for (VmStateHistoryEntry entry : vm.getStateHistory()) {
                if (previousTime != -1) {
                    double timeDiff = entry.getTime() - previousTime;
                    vmTotalAllocated += previousAllocated * timeDiff;
                    vmTotalRequested += previousRequested * timeDiff;

                    if (previousAllocated < previousRequested) {
                        slaViolation.add((previousRequested - previousAllocated) / previousRequested);
                        if (previousIsInMigration) {
                            vmUnderAllocatedDueToMigration += (previousRequested - previousAllocated)
                                    * timeDiff;
                        }
                    }
                }

                previousAllocated = entry.getAllocatedMips();
                previousRequested = entry.getRequestedMips();
                previousTime = entry.getTime();
                previousIsInMigration = entry.isInMigration();
            }

            totalAllocated += vmTotalAllocated;
            totalRequested += vmTotalRequested;
        }

        return (totalRequested - totalAllocated) / totalRequested;
    }
}
