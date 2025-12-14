{
  description = "Dev environment for slam-rs";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        llvmPackages = pkgs.llvmPackages_19;
        # OpenCV with GUI support (GTK)
        opencvWithGtk = pkgs.opencv.override {
          enableGtk3 = true;
          enableFfmpeg = true;
        };
      in
      {
        devShell = pkgs.mkShell {
          packages = [
            pkgs.rustup
            pkgs.pkg-config
            opencvWithGtk

            # GTK dep for OpenCV GUI
            pkgs.gtk3
            pkgs.glib
            pkgs.pango
            pkgs.cairo
            pkgs.gdk-pixbuf
            pkgs.atk

            pkgs.rerun

            # LLVM/Clang toolchain
            llvmPackages.clang
            llvmPackages.libclang

            pkgs.cmake

            # Python with uv for environment management
            pkgs.python3
            pkgs.uv

            # For tch (PyTorch bindings)
            pkgs.libtorch-bin

            # FFmpeg for video support
            pkgs.ffmpeg
          ];

          OPENCV_INCLUDE_PATH = "${opencvWithGtk}/include";
          OPENCV_LINK_PATH    = "${opencvWithGtk}/lib";
          PKG_CONFIG_PATH     = "${opencvWithGtk}/lib/pkgconfig:${pkgs.gtk3}/lib/pkgconfig:${pkgs.glib.dev}/lib/pkgconfig";

          LIBCLANG_PATH = "${llvmPackages.libclang.lib}/lib";

          # For tch/libtorch - telling torch-sys to use the system libtorch
          LIBTORCH = "${pkgs.libtorch-bin}";
          LIBTORCH_LIB = "${pkgs.libtorch-bin}/lib";
          LIBTORCH_INCLUDE = "${pkgs.libtorch-bin}/include";
          LIBTORCH_USE_PYTORCH = "0";
          LIBTORCH_BYPASS_VERSION_CHECK = "1";
          LIBTORCH_CXX11_ABI = "1";

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
            pkgs.libtorch-bin
            llvmPackages.libclang.lib
            opencvWithGtk
            pkgs.gtk3
            pkgs.glib
            pkgs.pango
            pkgs.cairo
            pkgs.gdk-pixbuf
            pkgs.atk
            pkgs.ffmpeg
          ];

          CC = "${llvmPackages.clang}/bin/clang";
          CXX = "${llvmPackages.clang}/bin/clang++";

          shellHook = ''
            rustup toolchain install stable --profile minimal >/dev/null 2>&1 || true
            rustup default stable
            echo ""
          '';
        };
      }
    );
}
