package org.cloudbus.cloudsim.provisioners;

import org.cloudbus.cloudsim.Vm;

import java.util.HashMap;
import java.util.Map;

public class DiskBwProvisionerSimple extends DiskBwProvisioner {

    /** The DiskBw map, where each key is a VM id and each value
     * is the amount of DiskBw allocated to that VM. */
    private Map<String, Long> DiskBwTable;

    /**
     * Instantiates a new DiskBw provisioner simple.
     *
     * @param DiskBw The total DiskBw capacity from the host that the provisioner can allocate to VMs.
     */
    public DiskBwProvisionerSimple(long DiskBw) {
        super(DiskBw);
        setDiskBwTable(new HashMap<String, Long>());
    }

    @Override
    public boolean allocateDiskBwForVm(Vm vm, long DiskBw) {
        deallocateDiskBwForVm(vm);

        if (getAvailableDiskBw() >= DiskBw) {
            setAvailableDiskBw(getAvailableDiskBw() - DiskBw);
            getDiskBwTable().put(vm.getUid(), DiskBw);
            vm.setCurrentAllocatedDiskBw(getAllocatedDiskBwForVm(vm));
            return true;
        }

        vm.setCurrentAllocatedDiskBw(getAllocatedDiskBwForVm(vm));
        return false;
    }

    @Override
    public long getAllocatedDiskBwForVm(Vm vm) {
        if (getDiskBwTable().containsKey(vm.getUid())) {
            return getDiskBwTable().get(vm.getUid());
        }
        return 0;
    }

    @Override
    public void deallocateDiskBwForVm(Vm vm) {
        if (getDiskBwTable().containsKey(vm.getUid())) {
            long amountFreed = getDiskBwTable().remove(vm.getUid());
            setAvailableDiskBw(getAvailableDiskBw() + amountFreed);
            vm.setCurrentAllocatedDiskBw(0);
        }
    }

    @Override
    public void deallocateDiskBwForAllVms() {
        super.deallocateDiskBwForAllVms();
        getDiskBwTable().clear();
    }

    @Override
    public boolean isSuitableForVm(Vm vm, long DiskBw) {
        long allocatedDiskBw = getAllocatedDiskBwForVm(vm);
        boolean result = allocateDiskBwForVm(vm, DiskBw);
        deallocateDiskBwForVm(vm);
        if (allocatedDiskBw > 0) {
            allocateDiskBwForVm(vm, allocatedDiskBw);
        }
        return result;
    }

    /**
     * Gets the map between VMs and allocated DiskBw.
     *
     * @return the DiskBw map
     */
    protected Map<String, Long> getDiskBwTable() {
        return DiskBwTable;
    }

    /**
     * Sets the map between VMs and allocated DiskBw.
     *
     * @param DiskBwTable the DiskBw map
     */
    protected void setDiskBwTable(Map<String, Long> DiskBwTable) {
        this.DiskBwTable = DiskBwTable;
    }

}
