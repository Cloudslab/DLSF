package org.cloudbus.cloudsim.power;

import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.core.CloudSimTags;
import org.cloudbus.cloudsim.core.predicates.PredicateType;
import org.cloudbus.cloudsim.plus.util.CustomLog;
import org.cloudbus.cloudsim.util.MathUtil;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.charset.Charset;
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

    private double totalResponseTime = 0;

    private double totalMigrationTime = 0;

    private double totalCompletionTime = 0;

    private int InputLimit = 100;

    private int lastMigrationCount = 0;

    private PowerVmAllocationPolicyMigrationStaticThreshold vmAllocSt = new PowerVmAllocationPolicyMigrationStaticThreshold(this.getHostList(),null,0.7);

    private PowerVmAllocationPolicyMigrationLocalRegression vmAllocLr = new PowerVmAllocationPolicyMigrationLocalRegression(this.getHostList(),null,0,300,vmAllocSt);

    private PowerVmAllocationPolicyMigrationMedianAbsoluteDeviation vmAllocMad = new PowerVmAllocationPolicyMigrationMedianAbsoluteDeviation(this.getHostList(),null,0,vmAllocSt);

    private PowerVmSelectionPolicyMaximumCorrelation vmSelMc = new PowerVmSelectionPolicyMaximumCorrelation(new PowerVmSelectionPolicyMinimumMigrationTime());

    private PowerVmSelectionPolicyMinimumMigrationTime vmSelMmt = new PowerVmSelectionPolicyMinimumMigrationTime();

    private PowerVmSelectionPolicyMinimumUtilization vmSelMu = new PowerVmSelectionPolicyMinimumUtilization();

    /** Python Interpreter
     * for interaction with DL code
     */
    protected static Process pythonProc;

    // get an outputstream to write into the standard input of python
    protected static PrintStream toPython;

    // get an inputstream to read from the standard output of python
    protected  static BufferedReader fromPython;

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
            DRLDatacenterBroker broker,
            String execFile) throws Exception {
        super(name, characteristics, vmAllocationPolicy, storageList, schedulingInterval);
        this.broker = broker;
        this.hostEnergy = new double[this.getHostList().size()];
        try{
            ProcessBuilder pb = new ProcessBuilder("python",execFile);
            pythonProc = pb.start();
        }
        catch(Exception e){System.out.println(e.getMessage());}
        System.out.println("Ran python code : " + execFile);
        toPython = new PrintStream(pythonProc.getOutputStream());
        fromPython = new BufferedReader(new InputStreamReader(pythonProc.getInputStream(), Charset.defaultCharset()));
    }

    protected void updateDLModel(){
        String loss = "Not learning!";
        try{
            if(getVmAllocationPolicy().getClass().getName().equals("org.cloudbus.cloudsim.power.DRLVmAllocationPolicy")){
//                toPython.println("backprop\n"+getLoss()+"END"); toPython.flush();
            }
            else{
//                toPython.println(("sendMap\n"+getInputMap()+"END")); toPython.flush();
            }
//            loss = DRLDatacenter.fromPython.readLine();
            System.out.println("DL Loss = " + loss);
            toPython.println(("setInput\n"+getInput()+"END")); toPython.flush();
        }
        catch(Exception e) {
            System.out.println(e.getMessage());
        }
    }

    public String getLoss(){
        String loss = "";
        double totalDataCenterEnergy = 0.0;
        double totalDataCenterCost = 0.0;
        for(DRLHost host : this.<DRLHost>getHostList()){
            totalDataCenterEnergy += this.hostEnergy[getHostList().indexOf(host)];
            totalDataCenterCost += (host.getCostModel().getCostPerCPUtime() * this.savedTimeDiff) / (60 * 60);
        }
        loss = loss + "CurrentTime " + this.savedCurrentTime +  "\n";
        loss = loss + "LastTime " + this.savedLastTime +  "\n";
        loss = loss + "TimeDiff " + this.savedTimeDiff +  "\n";
        loss = loss + "TotalEnergy " + totalDataCenterEnergy +  "\n";
        loss = loss + "NumVsEnded " + this.numVmsEnded +  "\n";
        loss = loss + "AverageResponseTime " + this.totalResponseTime/this.numVmsEnded +  "\n";
        loss = loss + "AverageMigrationTime " + this.totalMigrationTime/this.numVmsEnded +  "\n";
        loss = loss + "AverageCompletionTime " + this.totalCompletionTime/this.numVmsEnded + "\n";
        this.numVmsEnded = 0; this.totalMigrationTime = 0; this.totalResponseTime = 0;
        loss = loss + "TotalCost " + totalDataCenterCost +  "\n";
        loss = loss + "SLAOverall " + getSlaOverall(this.getVmList()) +  "\n";
        loss = loss + "VMsMigrated " + (this.getMigrationCount() - this.lastMigrationCount) + "\n";
        this.lastMigrationCount = this.getMigrationCount();
        if(getVmAllocationPolicy().getClass().getName().equals("org.cloudbus.cloudsim.power.DRLVmAllocationPolicy")){
            loss = loss + "HostPenalty " + ((DRLVmAllocationPolicy) getVmAllocationPolicy()).hostPenalty +  "\n";
            loss = loss + "MigrationPenalty " + ((DRLVmSelectionPolicy)
                    ((DRLVmAllocationPolicy)
                            getVmAllocationPolicy()).getVmSelectionPolicy()).migrationPenalty +  "\n";
        }
        return loss;
    }

    public String getInput(){
        String input = "";
        input = input + "CNN " + "number of VMs " + this.getVmList().size() +  "\n";
        String temp;
        for(PowerVm vm : this.<PowerVm>getVmList()){
            temp = "";
            PowerHost host = (PowerHost)vm.getHost();
            temp = temp + ((host != null) ? oneHot(this.getHostList().indexOf(host), 100) : "NA")  + " "; // Categorical
            temp = temp + vm.getNumberOfPes() + " "; // Continuous
            temp = temp + MathUtil.sum(vm.getCurrentRequestedMips()) + " "; // Continuous
            temp = temp + vm.getCurrentRequestedMaxMips() + " "; // Continuous
            temp = temp + vm.getUtilizationMean() + " "; // Continuous
            temp = temp + vm.getUtilizationVariance() + " "; // Continuous
            temp = temp + vm.getSize() + " "; // Continuous
            temp = temp + vm.getCurrentAllocatedSize() + " "; // Continuous
            temp = temp + vm.getRam() + " "; // Continuous
            temp = temp + vm.getCurrentAllocatedRam() + " "; // Continuous
            temp = temp + vm.getCurrentRequestedRam() + " "; // Continuous
            temp = temp + vm.getBw() + " "; // Continuous
            temp = temp + vm.getCurrentAllocatedBw() + " "; // Continuous
            temp = temp + vm.getCurrentRequestedBw() + " "; // Continuous
            temp = temp + vm.getDiskBw() + " "; // Continuous
            temp = temp + vm.getCurrentAllocatedDiskBw() + " "; // Continuous
            temp = temp + vm.getCurrentRequestedDiskBw() + " "; // Continuous
            temp = temp + vm.isInMigration() + " "; // Continuous
            ArrayList<Vm> list = new ArrayList<Vm>(Arrays.asList(vm));
            temp = temp + getSlaOverall(list) + " "; // Continuous
            temp = temp + ((host != null) ? (host.getUtilizationOfCpuMips()) : "NA")  + " "; // Continuous
            temp = temp + ((host != null) ? (host.getAvailableMips()) : "NA")  + " "; // Continuous
            temp = temp + ((host != null) ? (host.getRamProvisioner().getAvailableRam()) : "NA")  + " "; // Continuous
            temp = temp + ((host != null) ? (host.getBwProvisioner().getAvailableBw()) : "NA")  + " "; // Continuous
            temp = temp + ((host != null) ? (host.getDiskBwProvisioner().getAvailableDiskBw()) : "NA")  + " "; // Continuous
            temp = temp + ((host != null) ? (host.getVmList().size()) : "0")  + " "; // Continuous
            temp = temp + ((host != null) ? (this.hostEnergy[getHostList().indexOf(vm.getHost())]) : "NA") + " "; // Continuous
            temp = temp + ((host != null) ? (((double)vm.getRam())/vm.getHost().getBw()) : "NA"); // Continuous
            input = input + temp +  "\n";
        }
        input = input + "LSTM data\n";
        for(DRLHost host : this.<DRLHost>getHostList()){
            temp = "";
            temp = temp + this.hostEnergy[getHostList().indexOf(host)] + " "; // Continuous
            temp = temp + host.getPower() + " "; // Continuous
            temp = temp + host.getMaxPower() + " "; // Continuous
            temp = temp + (host.getCostModel().getCostPerCPUtime() * this.savedTimeDiff / (60 * 60)) + " "; // Continuous
            temp = temp + getSlaOverall(host.getVmList()) + " "; // Continuous
            temp = temp + host.getUtilizationOfCpu() + " "; // Continuous
            temp = temp + host.getMaxUtilization() + " "; // Continuous
            temp = temp + host.getNumberOfPes() + " "; // Continuous
            temp = temp + host.getRamProvisioner().getAvailableRam() + " "; // Continuous
            temp = temp + host.getRamProvisioner().getRam() + " "; // Continuous
            temp = temp + host.getBwProvisioner().getAvailableBw() + " "; // Continuous
            temp = temp + host.getBwProvisioner().getBw() + " "; // Continuous
            temp = temp + host.getDiskBwProvisioner().getAvailableDiskBw() + " "; // Continuous
            temp = temp + host.getDiskBwProvisioner().getDiskBw() + " "; // Continuous
            temp = temp + host.getVmsMigratingIn().size() + " "; // Continuous
            // Parameters from other policies
//            temp = temp + vmAllocLr.isHostOverUtilized(host) + " "; // Boolean
            temp = temp + vmAllocSt.isHostOverUtilized(host) + " "; // Boolean
//            temp = temp + vmAllocMad.isHostOverUtilized(host) + " "; // Boolean
//            temp = temp + oneHot(this.getVmList().indexOf(vmSelMc.getVmToMigrate(host)),100) + " "; // Categorical
//            temp = temp + oneHot(this.getVmList().indexOf(vmSelMmt.getVmToMigrate(host)),100) + " "; // Categorical
            temp = temp + oneHot(this.getVmList().indexOf(vmSelMu.getVmToMigrate(host)),100); // Categorical
            input = input + temp +  "\n";
        }
        return input;
    }

    public String oneHot(int value, int range){
        String res = "";
        for(int i = 0; i < range; i++){
            res = res + ((value == i) ? "1" : "0");
            if(i < range - 1)
                res += " ";
        }
        return res;
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
        if(this.savedTimeDiff > 200){ // getVmAllocationPolicy().getClass().getName().equals("DRLVmAllocationPolicy") &&
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
            map = map + (this.getVmList().indexOf(vm) + " " + vm.getHost().getId()) +  "\n";
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
        this.totalResponseTime += ((DRLHost)((DRLVm)vm).getHost()).getResponseTime();
        this.totalMigrationTime += ((DRLVm)vm).totalMigrationTime;
        this.totalCompletionTime += (CloudSim.clock() - ((DRLVm)vm).startTime);
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
