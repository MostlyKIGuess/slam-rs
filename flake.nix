{
  description = "Dev environment for slam-rs with OpenCV, Rerun, nalgebra, etc.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.mkFlake {
      systems = flake-utils.defaultSystems;

      perSystem = { system }:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in {
          devShells.default = pkgs.mkShell {
            packages = [
              # Rust toolchain
              pkgs.rustup
              pkgs.pkg-config

              # OpenCV (system-wide libs)
              pkgs.opencv

              # rerun-cli (viewer)
              pkgs.rerun-cli

              # Optional helper tools
              pkgs.clang
              pkgs.cmake
            ];

            # Let Rust find OpenCV headers + libs
            OPENCV_INCLUDE_PATH = "${pkgs.opencv}/include";
            OPENCV_LINK_PATH = "${pkgs.opencv}/lib";

            # For pkg-config
            PKG_CONFIG_PATH = "${pkgs.opencv}/lib/pkgconfig";

            # Make cargo use clang (helps OpenCV build issues)
            CC = "clang";
            CXX = "clang++";

            # Enable rustup toolchain loading
            shellHook = ''
              rustup toolchain install stable --profile minimal >/dev/null 2>&1 || true
              rustup default stable
            '';
          };
        };
    };
}
