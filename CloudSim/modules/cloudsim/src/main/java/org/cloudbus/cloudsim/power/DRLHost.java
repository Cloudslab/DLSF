package org.cloudbus.cloudsim.power;

import org.cloudbus.cloudsim.*;
import org.cloudbus.cloudsim.core.CloudSim;
import org.cloudbus.cloudsim.cost.models.CostModel;
import org.cloudbus.cloudsim.lists.PeList;
import org.cloudbus.cloudsim.power.models.PowerModel;
import org.cloudbus.cloudsim.provisioners.BwProvisioner;
import org.cloudbus.cloudsim.provisioners.DiskBwProvisioner;
import org.cloudbus.cloudsim.provisioners.RamProvisioner;
import org.cloudbus.cloudsim.util.MathUtil;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class DRLHost extends PowerHostUtilizationHistory{

    /** The utilization mips. */
    private double utilizationMips;

    /** The previous utilization mips. */
    private double previousUtilizationMips;

    /** The host utilization state history. */
    private final List<HostStateHistoryEntry> stateHistory = new LinkedList<HostStateHistoryEntry>();

    /** Cost Model of this host **/
    private CostModel costModel;

    /** Fog node **/
    public boolean isFog = false;

    /** Response time of Fog node in seconds **/
    private double fogResponseTime = 0.001;

    /** Response time of Fog node in seconds **/
    private double cloudResponseTime = 0.01;

    /**
     * Instantiates a new PowerHost.
     *
     * @param id the id of the host
     * @param ramProvisioner the ram provisioner
     * @param bwProvisioner the bw provisioner
     * @param storage the storage capacity
     * @param peList the host's PEs list
     * @param vmScheduler the VM scheduler
     * @param powerModel the power model of host
     */
    public DRLHost(
            int id,
            RamProvisioner ramProvisioner,
            BwProvisioner bwProvisioner,
            DiskBwProvisioner diskBwProvisioner,
            long storage,
            List<? extends Pe> peList,
            VmScheduler vmScheduler,
            PowerModel powerModel,
            CostModel costModel,
            boolean fog) {
        super(id, ramProvisioner, bwProvisioner, storage, peList, vmScheduler, powerModel);
        setDiskBwProvisioner(diskBwProvisioner);
        setCostModel(costModel);
        this.isFog = fog;
    }

    /**
     * Get cost model of this host
     *
     * @return cost model
     */
    protected CostModel getCostModel(){
        return costModel;
    }

    /**
     * Set cost model of this host
     *
     * @param cm cost model to set
     */
    protected void setCostModel(CostModel cm){
        this.costModel = cm;
    }

    /** Host Dynamic Workload mehtods **/
    public double updateVmsProcessing(double currentTime) {
        Log.setDisabled(true);
        double smallerTime = super.updateVmsProcessing(currentTime);
        Log.setDisabled(false);
        setPreviousUtilizationMips(getUtilizationMips());
        setUtilizationMips(0);
        double hostTotalRequestedMips = 0;

        for (Vm vm : getVmList()) {
            getVmScheduler().deallocatePesForVm(vm);
        }

        for (Vm vm : getVmList()) {
            getVmScheduler().allocatePesForVm(vm, vm.getCurrentRequestedMips());
        }

        for (Vm vm : getVmList()) {
            double totalRequestedMips = vm.getCurrentRequestedTotalMips();
            double totalAllocatedMips = getVmScheduler().getTotalAllocatedMipsForVm(vm);

            if (!Log.isDisabled()) {
                try {
                    Log.formatLine(
                            "%.2f: [Host #" + getId() + "] Total allocated MIPS for VM #" + vm.getId()
                                    + " (Host #" + vm.getHost().getId()
                                    + ") is %.2f, was requested %.2f out of total %.2f (%.2f%%)",
                            CloudSim.clock(),
                            totalAllocatedMips,
                            totalRequestedMips,
                            vm.getMips(),
                            totalRequestedMips / vm.getMips() * 100);
                }
                catch (Exception e){}

                List<Pe> pes = getVmScheduler().getPesAllocatedForVM(vm);
                StringBuilder pesString = new StringBuilder();
                for (Pe pe : pes) {
                    pesString.append(String.format(" PE #" + pe.getId() + ": %.2f.", pe.getPeProvisioner()
                            .getTotalAllocatedMipsForVm(vm)));
                }
                Log.formatLine(
                        "%.2f: [Host #" + getId() + "] MIPS for VM #" + vm.getId() + " by PEs ("
                                + getNumberOfPes() + " * " + getVmScheduler().getPeCapacity() + ")."
                                + pesString,
                        CloudSim.clock());
            }

            if (getVmsMigratingIn().contains(vm)) {
                Log.formatLine("%.2f: [Host #" + getId() + "] VM #" + vm.getId()
                        + " is being migrated to Host #" + getId(), CloudSim.clock());
            } else {
                if (totalAllocatedMips + 0.1 < totalRequestedMips) {
                    Log.formatLine("%.2f: [Host #" + getId() + "] Under allocated MIPS for VM #" + vm.getId()
                            + ": %.2f", CloudSim.clock(), totalRequestedMips - totalAllocatedMips);
                }

                vm.addStateHistoryEntry(
                        currentTime,
                        totalAllocatedMips,
                        totalRequestedMips,
                        (vm.isInMigration() && !getVmsMigratingIn().contains(vm)));

                if (vm.isInMigration()) {
                    Log.formatLine(
                            "%.2f: [Host #" + getId() + "] VM #" + vm.getId() + " is in migration",
                            CloudSim.clock());
                    totalAllocatedMips /= 0.9; // performance degradation due to migration - 10%
                }
            }

            setUtilizationMips(getUtilizationMips() + totalAllocatedMips);
            hostTotalRequestedMips += totalRequestedMips;
        }

        addStateHistoryEntry(
                currentTime,
                getUtilizationMips(),
                hostTotalRequestedMips,
                (getUtilizationMips() > 0));

        return smallerTime;
    }

    /**
     * Gets the list of completed vms.
     *
     * @return the completed vms
     */
    public List<Vm> getCompletedVms() {
        List<Vm> vmsToRemove = new ArrayList<Vm>();
        for (Vm vm : getVmList()) {
            if (vm.isInMigration()) {
                continue;
            }
            if (vm.getCurrentRequestedTotalMips() == 0) {
                vmsToRemove.add(vm);
            }
        }
        return vmsToRemove;
    }

    /**
     * Gets the max utilization percentage among by all PEs.
     *
     * @return the maximum utilization percentage
     */
    public double getMaxUtilization() {
        return PeList.getMaxUtilization(getPeList());
    }

    /**
     * Gets the max utilization percentage among by all PEs allocated to a VM.
     *
     * @param vm the vm
     * @return the max utilization percentage of the VM
     */
    public double getMaxUtilizationAmongVmsPes(Vm vm) {
        return PeList.getMaxUtilizationAmongVmsPes(getPeList(), vm);
    }

    /**
     * Gets the utilization of memory (in absolute values).
     *
     * @return the utilization of memory
     */
    public double getUtilizationOfRam() {
        return getRamProvisioner().getUsedRam();
    }

    /**
     * Gets the utilization of bw (in absolute values).
     *
     * @return the utilization of bw
     */
    public double getUtilizationOfBw() {
        return getBwProvisioner().getUsedBw();
    }

    /**
     * Gets the utilization of bw (in absolute values).
     *
     * @return the utilization of bw
     */
    public double getUtilizationOfDiskBw() {
        return getDiskBwProvisioner().getUsedDiskBw();
    }

    /**
     * Get current utilization of CPU in percentage.
     *
     * @return current utilization of CPU in percents
     */
    public double getUtilizationOfCpu() {
        double utilization = getUtilizationMips() / getTotalMips();
        if (utilization > 1 && utilization < 1.01) {
            utilization = 1;
        }
        return utilization;
    }

    /**
     * Gets the previous utilization of CPU in percentage.
     *
     * @return the previous utilization of cpu in percents
     */
    public double getPreviousUtilizationOfCpu() {
        double utilization = getPreviousUtilizationMips() / getTotalMips();
        if (utilization > 1 && utilization < 1.01) {
            utilization = 1;
        }
        return utilization;
    }

    /**
     * Get current utilization of CPU in MIPS.
     *
     * @return current utilization of CPU in MIPS
     * @todo This method only calls the  {@link #getUtilizationMips()}.
     * getUtilizationMips may be deprecated and its code copied here.
     */
    public double getUtilizationOfCpuMips() {
        return getUtilizationMips();
    }

    /**
     * Gets the utilization of CPU in MIPS.
     *
     * @return current utilization of CPU in MIPS
     */
    public double getUtilizationMips() {
        return utilizationMips;
    }

    /**
     * Sets the utilization mips.
     *
     * @param utilizationMips the new utilization mips
     */
    protected void setUtilizationMips(double utilizationMips) {
        this.utilizationMips = utilizationMips;
    }

    /**
     * Gets the previous utilization of CPU in mips.
     *
     * @return the previous utilization of CPU in mips
     */
    public double getPreviousUtilizationMips() {
        return previousUtilizationMips;
    }

    /**
     * Sets the previous utilization of CPU in mips.
     *
     * @param previousUtilizationMips the new previous utilization of CPU in mips
     */
    protected void setPreviousUtilizationMips(double previousUtilizationMips) {
        this.previousUtilizationMips = previousUtilizationMips;
    }

    /**
     * Gets the host state history.
     *
     * @return the state history
     */
    public List<HostStateHistoryEntry> getStateHistory() {
        return stateHistory;
    }

    /**
     * Gets response time based on whether the node is fog or cloud
     *
     * @return the response time in seconds
     */
    public double getResponseTime(){return (isFog ? fogResponseTime : cloudResponseTime);}

    /**
     * Adds a host state history entry.
     *
     * @param time the time
     * @param allocatedMips the allocated mips
     * @param requestedMips the requested mips
     * @param isActive the is active
     */
    public
    void
    addStateHistoryEntry(double time, double allocatedMips, double requestedMips, boolean isActive) {

        HostStateHistoryEntry newState = new HostStateHistoryEntry(
                time,
                allocatedMips,
                requestedMips,
                isActive);
        if (!getStateHistory().isEmpty()) {
            HostStateHistoryEntry previousState = getStateHistory().get(getStateHistory().size() - 1);
            if (previousState.getTime() == time) {
                getStateHistory().set(getStateHistory().size() - 1, newState);
                return;
            }
        }
        getStateHistory().add(newState);
    }
}
