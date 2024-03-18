    def spectral_pooling(self, x, pooling_number = 0):
        '''
        Carry out a spectral pooling.
        torch.rfft(x, signal_ndim, normalized, onesided)
        signal_ndim takes into account the signal_ndim dimensions stranting from the last one
        onesided if True, outputs only the positives frequencies, under the nyquist frequency

        @param x: input sequence
        @return x: output of spectral pooling
        '''
        # xpool = F.max_pool2d(x, (2, 1))

        x = x.permute(0, 1, 3, 2)

        # plt.figure()
        # f, axarr = plt.subplots(5, 1)

        # x_plt = x[0, 0].to("cpu", torch.double).detach()
        # axarr[0].plot(x_plt[0], label='input')

        #fft = torch.rfft(x, signal_ndim=1, normalized=True, onesided=True)
        fft = torch.fft.rfft(x, norm="forward")
        if self.config["storing_acts"]:
            self.save_acts(fft, "x_LA_fft")
        # fft2 = torch.rfft(x, signal_ndim=1, normalized=False, onesided=False)

        # fft_plt = fft[0, 0].to("cpu", torch.double).detach()
        # fft_plt = torch.norm(fft_plt, dim=2)
        # axarr[1].plot(fft_plt[0], 'o', label='fft')

        #x = fft[:, :, :, :int(fft.shape[3] / 2)]
        x = fft[:, :, :, :int(math.ceil(fft.shape[3] / 2))]
        if self.config["storing_acts"]:
            self.save_acts(x, "x_LA_fft_2")

        # fftx_plt = x[0, 0].to("cpu", torch.double).detach()
        # fftx_plt = torch.norm(fftx_plt, dim=2)
        # axarr[2].plot(fftx_plt[0], 'o', label='fft')

        # x = torch.irfft(x, signal_ndim=1, normalized=True, onesided=True)
        x = torch.fft.irfft(x, norm="forward")
        if self.config["storing_acts"]:
            self.save_acts(x, "x_LA_ifft")

        x = x[:, :, :, :self.pooling_Wx[pooling_number]]
        if self.config["storing_acts"]:
            self.save_acts(x, "x_LA_ifft_pool")

        # x_plt = x[0, 0].to("cpu", torch.double).detach()
        # axarr[3].plot(x_plt[0], label='input')

        x = x.permute(0, 1, 3, 2)

        # fft2_plt = fft2[0, 0].to("cpu", torch.double).detach()
        # fft2_plt = torch.norm(fft2_plt, dim=2)
        # print(fft2_plt.size(), 'max: {}'.format(torch.max(fft2_plt)), 'min: {}'.format(torch.min(fft2_plt)))
        # axarr[4].plot(fft2_plt[0], 'o', label='fft')

        # xpool = xpool.permute(0, 1, 3, 2)
        # x_plt = xpool[0, 0].to("cpu", torch.double).detach()
        # axarr[3].plot(x_plt[0], label='input')

        # plt.waitforbuttonpress(0)
        # plt.close()


        return x