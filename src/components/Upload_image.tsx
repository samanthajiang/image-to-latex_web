import React, { useState, useRef } from 'react';

import CloseIcon from '@mui/icons-material/Close';
import { styled } from '@mui/material/styles';
import Switch from '@mui/material/Switch';
import axios from 'axios';

import EditableCode from './editableCode';
import LoadingSpinner from './Spinner';
import 'katex/dist/katex.min.css';

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
  const [imageSrc, setImageSrc] = useState('');
  const [imgName, setImgName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [loading, setLoading] = React.useState(false);
  const modelState = useRef(false);
  function handleClick() {
    setImageSrc('');
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
      if (modelState.current) {
        const response = await axios.post(
          'http://127.0.0.1:8001/predict_old/',
          formData,
          requestConfig
        );
        setIsLoading(false);
        setImgName(response.data.prediction);
        setLoading(false);
      } else {
        const response = await axios.post(
          'http://127.0.0.1:8001/predict/',
          formData,
          requestConfig
        );
        setIsLoading(false);
        setImgName(response.data.prediction);
        setLoading(false);
      }

      // const response = await axios.post(
      //   'http://127.0.0.1:8001/predict/',
      //   formData,
      //   requestConfig
      // );
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
      // eslint-disable-next-line no-alert
      alert('Please select a valid image file.');
      setImageSrc('');
    }
  };

  const modelChange = () => {
    modelState.current = !modelState.current;
    check();
  };

  return (
    <div className={`container max-w-5xl mx-auto m-12`}>
      <div className="mt-5 sm:mt-8 sm:flex sm:justify-center lg:justify-start">
        <div id="drop-area">
          <VisuallyHiddenInput
            type="file"
            id="fileElem"
            accept="image/*"
            // onChange={(e) => {
            //   this.handleFiles(e.target.files);
            // }}
          />
          {/* <label className="upload-label " htmlFor="fileElem"> */}
          {/*  <div className="upload-text ">Drag Image here or click to upload</div> */}
          {/* </label> */}
          <div className="image" />
        </div>
        {/*= ========= Drag ========== */}
        <div className="flex items-center justify-center w-full">
          <label
            htmlFor="dropzone-file"
            onChange={(e) => handleImageChange(e)}
            className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:hover:bg-bray-800 dark:bg-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:hover:border-gray-500 dark:hover:bg-gray-600"
          >
            <div className="flex flex-col items-center justify-center pt-5 pb-6">
              <svg
                className="w-10 h-10 mb-3 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                ></path>
              </svg>
              <p className="mb-2 text-xl text-primary dark:text-gray-400">
                <span className="font-semibold">Click to upload</span> or drag
                and drop
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                SVG, PNG, JPG or GIF (MAX. 2000x2000px)
              </p>
            </div>
            <input id="dropzone-file" type="file" className="hidden" />
          </label>
        </div>
      </div>
      <div className="mt-5 sm:mt-8 sm:flex sm:justify-center lg:justify-start">
        {imageSrc ? (
          <div className="mx-auto flex items-center ...">
            <div className="mx-8 my-6 ...">
              <img srcSet={URL.createObjectURL(imageSrc)} alt="Image" />
            </div>
            <div className="mr:2 ...">
              <CloseIcon className="hover:bg-gray-200" onClick={handleClick} />
            </div>
          </div>
        ) : (
          <></>
        )}
      </div>

      <div className="convert flex flex-row items-center...">
        <div>
          <button
            disabled={loading}
            className="w-42 h-18.5 me-5 mb-5 flex items-center justify-center border border-transparent text-base font-medium rounded-md text-background bg-primary hover:bg-red-200 hover:border-transparent hover:text-gray-900 md:py-4 md:text-lg md:px-10 dark:hover:text-white"
            onClick={() => check()}
          >
            {loading ? (
              <span>LOADING...</span>
            ) : (
              <span style={{ whiteSpace: 'nowrap' }}>CONVERT</span>
            )}
          </button>
        </div>
        <div>
          <p
            className={
              'pt-4 pl-6 text-xl text-gray-500 lg:mx-auto text-gray-600'
            }
          >
            Unhappy about the result? Try{' '}
            <span className="text-primary">CONVERT</span> Again or{' '}
            <span className="text-primary">SWITCH</span> model{'     '}
            <Switch onChange={() => modelChange()} />
          </p>
        </div>
      </div>

      <div className="prediction">
        <h3 className="py-1 pl-4 text-xl text-gray-500 lg:mx-auto text-gray-600">
          <span className="text-primary">EDIT</span> the Generated Latex formula
          For Better Result
        </h3>
        {isLoading ? (
          <LoadingSpinner />
        ) : (
          <div>
            <EditableCode value={imgName} />
          </div>
        )}
      </div>
    </div>
  );
};

export default Image;
