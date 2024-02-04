import React, { useState } from 'react';

import CloseIcon from '@mui/icons-material/Close';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import LoadingButton from '@mui/lab/LoadingButton';
import Button from '@mui/material/Button';
import { styled } from '@mui/material/styles';
import axios from 'axios';
import { BlockMath } from 'react-katex';
import Box from "@mui/material/Box";
import Katex from './Katex';
import LoadingSpinner from './Spinner';
import 'katex/dist/katex.min.css';
import EditableCode from "./editableCode";

// @ts-ignore

// eslint-disable-next-line @typescript-eslint/no-unused-expressions
('use client');
const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 100,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 100,
  left: 100,
  whiteSpace: 'nowrap',
  width: 1,
});

const Image = () => {
  const [imageSrc, setImageSrc] = useState(null);
  const [imgName, setImgName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  // eslint-disable-next-line unused-imports/no-unused-vars
  const [value, setValue] = React.useState('**Hello world!!!**');
  const [loading, setLoading] = React.useState(false);
  function handleClick() {
    setImageSrc(null);
    setImgName('');
    setLoading(false);
  }

  const requestConfig = {
    timeout: 60000,
  };

  const check = async () => {
    setLoading(true);
    try {
      setIsLoading(true);
      const formData = new FormData();
      formData.append('imageFile', imageSrc);

      const response = await axios.post(
        'http://127.0.0.1:8001/predict/',
        formData,
        requestConfig
      );

      setIsLoading(false);
      setImgName(response.data.prediction);
      setLoading(false);
    } catch (error) {
      setLoading(false);
      setIsLoading(false);
      console.log(error);
    }
  };

  // @ts-ignore
  const handleImageChange = (e) => {
    const selectedFile = e.target.files[0];

    if (selectedFile && selectedFile.type.startsWith('image/')) {
      setImageSrc(selectedFile);
    } else {
      alert('Please select a valid image file.');
      setImageSrc(null);
    }
  };

  return (
    <div className={`container max-w-5xl mx-auto m-12`}>

      <div className="mt-5 sm:mt-8 sm:flex sm:justify-center lg:justify-start">
        <div id="drop-area">
        <VisuallyHiddenInput
          type="file"
          id="fileElem"
          accept="image/*"
          onChange={e => {
            this.handleFiles(e.target.files);
          }}
        />
        {/*<label className="upload-label " htmlFor="fileElem">*/}
        {/*  <div className="upload-text ">Drag Image here or click to upload</div>*/}
        {/*</label>*/}
        <div className="image" />
      </div>
{/*========== Drag ==========*/}
        <div className="flex items-center justify-center w-full">
        <label htmlFor="dropzone-file" onChange={(e) => handleImageChange(e)}
               className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:hover:bg-bray-800 dark:bg-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:hover:border-gray-500 dark:hover:bg-gray-600">
          <div className="flex flex-col items-center justify-center pt-5 pb-6" >
            <svg className="w-10 h-10 mb-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"
                 xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2"
                    d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"></path>
            </svg>
            <p className="mb-2 text-xl text-primary dark:text-gray-400" ><span
                className="font-semibold">Click to upload</span> or drag and drop</p>
            <p className="text-xs text-gray-500 dark:text-gray-400">SVG, PNG, JPG or GIF (MAX. 2000x2000px)</p>
          </div>
          <input id="dropzone-file" type="file" className="hidden"/>
        </label>
      </div>


        {/*<div className="rounded-md shadow mx-auto">*/}
        {/*  <form>*/}

        {/*    <Button*/}
        {/*      className={`w-full flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-background bg-primary hover:bg-border hover:text-primary md:py-4 md:text-lg md:px-10`}*/}
        {/*      component="label"*/}
        {/*      variant="contained"*/}
        {/*      startIcon={<CloudUploadIcon />}*/}
        {/*    >*/}
        {/*      Upload Image*/}
        {/*      <VisuallyHiddenInput*/}
        {/*        type="file"*/}
        {/*        onChange={(e) => handleImageChange(e)}*/}
        {/*      />*/}
        {/*    </Button>*/}
        {/*  </form>*/}
        {/*</div>*/}
      </div>
      <div className="mt-5 sm:mt-8 sm:flex sm:justify-center lg:justify-start">

          {imageSrc ? (
            <div className="mx-auto flex items-center ...">
              <div className="mx-8 my-6 ...">
              <img srcSet={URL.createObjectURL(imageSrc)} alt="Image" />
            </div>
              <div className="mr:2 ...">
              <CloseIcon className='hover:bg-gray-200' onClick={handleClick} />
                </div>
              </div>
          ) : (
            <></>
          )}

      </div>

      <div className="convert flex flex-row items-center...">
        <div>
        <LoadingButton
          // className="py-2.5 px-8 me-5 mb-5 text-base font-medium text-gray-900 focus:outline-none bg-white rounded-lg border border-gray-600 hover:bg-gray-100 hover:text-blue-700 focus:z-10 focus:ring-4 focus:ring-gray-200 dark:focus:ring-gray-700 dark:bg-gray-800 dark:text-gray-400 dark:border-gray-600 dark:hover:text-white dark:hover:bg-gray-700"
          className = 'py-2.5 px-8 me-5 mb-5  flex items-center justify-center px-8 py-3 border border-transparent text-base font-medium rounded-md text-background bg-primary hover:bg-red-200 hover:border-transparent hover:text-gray-900 md:py-4 md:text-lg md:px-10 dark:hover:text-white'
          onClick={() => check()}
          loading={loading}
          loadingIndicator="Converting…"
          variant="outlined"
        >
          <span>Convert</span>
        </LoadingButton>
          </div>
        <div >
          <p className={'pt-4 pl-6 text-xl text-gray-500 lg:mx-auto text-gray-600'}>Unhappy about the result? Try <span className='text-primary'>CONVERT</span> Again </p>
        </div>
      </div>

      <div className="prediction">
        <h3 className='py-1 pl-4 text-xl text-gray-500 lg:mx-auto text-gray-600'>Edit the Generated Latex formula For Better Result</h3>
        {isLoading ? (
          <LoadingSpinner />
        ) : (
            <div>

            <EditableCode value = {imgName}/>
              </div>
        )}
      </div>
      {/*<div className={'bg-gray-100'}><BlockMath>{imgName}</BlockMath></div>*/}


    </div>
  );
};

export default Image;
